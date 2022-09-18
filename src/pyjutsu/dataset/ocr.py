"""
This package contains dataset helpers for working OCR-related subjects.
"""

import dataclasses
import itertools
import json
import logging
import os
import random
import shutil
import xml.dom.minidom
from pathlib import Path
from typing import List, Iterable, Union, Callable, Any, Dict, Sequence, Set
from xml.dom.minidom import Document, Element

import cv2
import numpy as np
from tqdm import tqdm

from pyjutsu.collection_utils import circular
from pyjutsu.image import SheetDimensions, augment_image, AUGThresholdMode, get_tiles
from pyjutsu.functional import apply_pipeline
from pyjutsu.geometry import Rect
from pyjutsu.json_serialization import to_jsonable, from_jsonable
from pyjutsu.os_utils import get_files, resolve_path
from pyjutsu.pydantic_model import NDArray, PydanticModel

GROUND_TRUTH_FILENAME_OLD = "gt.json"
GROUND_TRUTH_FILENAME = "ground-truth.json"
RANDOM_FORMS_PER_PAIR = 3

LOG = logging.getLogger(__name__)

class Word(PydanticModel):
    word_id: str
    text: str
    glyphs: List[Rect]

    def offset_x(self, x):
        for g in self.glyphs:
            g.x += x

    def get_top(self):
        return min(self.glyphs, key=lambda g: g.y).y

    def get_left(self):
        return min(self.glyphs, key=lambda g: g.x).x

    def get_right(self):
        max_x_r = max(self.glyphs, key=lambda g: g.x + g.width)
        return max_x_r.x + max_x_r.width

    def get_bottom(self):
        max_y_r = max(self.glyphs, key=lambda g: g.y + g.height)
        return max_y_r.y + max_y_r.height

    def get_rect(self):
        left, top, right, bottom = \
            self.get_left(), self.get_top(), self.get_right(), self.get_bottom()

        return Rect.from_xywh(left, top, right-left, bottom-top)

    def is_roi_valid(self):
        if not self.glyphs:
            return False

        r = self.get_rect()
        return bool(r.width and r.height)


class Line(PydanticModel):
    text: str
    words: List[Word]


HandwritingMarkup = List[Union[Line, Word]]


SAMPLE_SHEET_DEFAULT = SheetDimensions.get_a4()


class Sample(PydanticModel):
    sample_id: str
    markup: HandwritingMarkup
    img: Union[NDArray, Path]
    sheet: SheetDimensions = SAMPLE_SHEET_DEFAULT

    def dpi(self):
        return self.sheet.get_dpi_for(self.img)

    def as_stored(self, img_path: Path):
        c = self.copy()
        c.img = img_path
        return c


class DedicatedWord(Word):
    parent: Sample

    @staticmethod
    def from_word(w: Word, p: Sample):
        return DedicatedWord(parent=p, **dict(w))


class _StoredDataset(PydanticModel):
    """
    This is a root class for model we about to serialize/deserialize.
    To the user though it's much more convenient to obtain
    plain list of samples, and this is why we don't publish this class.
    """

    samples: Dict[str, Sample] = {}

    @staticmethod
    def from_samples(samples: Iterable[Sample]):
        return _StoredDataset(samples={
            sample.sample_id: sample for sample in samples
        })


def flatten(samples: Iterable[Sample]) -> Iterable[DedicatedWord]:
    """
    Flattens hierarchical page->line->word dataset into collection
    of dedicated words
    :param samples: list of dataset page samples
    :return: flattened collection of dedicated words.
    """
    for p in samples:
        for lnw in p.markup:
            if isinstance(lnw, Word):
                yield DedicatedWord.from_word(lnw, p)
                continue
            for w in lnw.words:
                yield DedicatedWord.from_word(w, p)


def _crop(img: np.ndarray, r: Rect):
    return img[r.y: r.y+r.height, r.x: r.x+r.width]


# TODO: cover with tests
def generate_dataset(
    forms: List[Sample],
    handwritings: List[Sample],
    min_words_per_form: int,
    max_words_per_form: int,
    tile_rows: int,
    tile_cols: int,
) -> Iterable[Sample]:
    """
    Generates dataset out of empty forms and cleansed handwritings.
    So far, it just puts handwriting in absolutely random way
    onto empty forms.
    Result is produced on-demand, with `yield` operator
    :param forms: collection of forms
    :param handwritings: cleansed images of handwritings (black lines with white background)
    :param min_words_per_form: minimum words per form.
    :param max_words_per_form: maximum words per form.
    :param tile_rows: amount of tiles per vertical dimension
    :param tile_cols: amount of tiles per horizontal dimension
    :return Collection of dataset samples
    """
    all_words_src: List[DedicatedWord] = [*flatten(handwritings)]

    if all_words_src:
        random.shuffle(all_words_src)

    # TODO: 50% of all forms should be copied empty
    all_words = circular(all_words_src) if all_words_src else None

    num_forms = len(forms)
    new_forms = list(forms)
    for i in range(RANDOM_FORMS_PER_PAIR):
        # Create form copies, each copy will have its own random contents
        new_forms += list(map(
            lambda form_sample: form_sample.copy(
                update=dict(sample_id=f"{form_sample.sample_id}_{i}")
            ),
            forms
        ))
    forms = new_forms

    random.shuffle(forms)
    forms = forms[:num_forms]

    sheet_dimensions: SheetDimensions = SheetDimensions.get_a4()

    def _random_crop(img):
        h, w = img.shape[:2]
        _left = random.randint(0, int(w*0.2))
        _top = random.randint(0, int(h*0.2))
        _right = random.randint(max(_left+1, int(w*0.8)), w)
        _bottom = random.randint(max(_top+1, int(h*0.8)), h)
        r = img[_top:_bottom, _left:_right]
        return r

    def _random_resize(img):
        sx = random.uniform(0.8, 1.2)
        sy = random.uniform(0.8, 1.2)
        return cv2.resize(img, dsize=None, fx=sx, fy=sy)

    def _threshold(img):
        res = augment_image(img, AUGThresholdMode.ONLY_THRESHOLD)
        return res["threshold"]

    def _gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _color(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def _random_color(img):

        img = np.copy(img)

        random_color = [*map(lambda x: random.randint(0, 128), [None]*3)]

        black_pixels = np.where(
            (img[:, :, 0] == 0) &
            (img[:, :, 1] == 0) &
            (img[:, :, 2] == 0)
        )

        # set those pixels to white
        img[black_pixels] = random_color

        return img

    def _gen_sample(sample_id, src_form, handwritten_words, form_resolution):
        new_form = np.copy(src_form)

        form_h, form_w = src_form.shape[:2]

        words = []
        for word in handwritten_words:
            def _change_dpi(img):
                scale = form_resolution / word.parent.dpi()
                return cv2.resize(img, dsize=None, fx=scale, fy=scale)

            w_rect = word.get_rect()
            hwi = _crop(word.parent.img, w_rect)
            hwi_aug = apply_pipeline(hwi, [
                _gray,
                _random_crop,
                _random_resize,
                _change_dpi,
                _threshold,
                _color,
                _random_color
            ])
            hwi_aug = _crop(
                hwi_aug,
                Rect.from_xywh(
                    0, 0,
                    min(form_w, w_rect.width), min(form_h, w_rect.height)
                )
            )

            hwi_h, hwi_w = hwi_aug.shape[:2]
            left = random.randint(0, form_w - hwi_w) if form_w > hwi_w else 0
            top = random.randint(0, form_h - hwi_h) if form_h > hwi_h else 0
            new_word_rect = Rect.from_xywh(left, top, hwi_w, hwi_h)
            form_place: np.ndarray = _crop(new_form, new_word_rect)
            form_place[:] = np.minimum(form_place, hwi_aug)
            words.append(Word(
                word_id=f"{sample_id}_{word.word_id}",
                text=word.text,
                glyphs=[new_word_rect]
            ))

        return Sample(sample_id=sample_id, markup=words, img=new_form)

    cur_word = 0
    for form_sample in forms:
        form_id = form_sample.sample_id
        form = form_sample.img
        original_resolution = sheet_dimensions.get_dpi_for(form)
        form_tiles = get_tiles(form, tile_rows, tile_cols)

        for form_tile_id, tile in enumerate(form_tiles):
            keep_empty = random.randint(0, 1)
            if keep_empty or not all_words:
                yield _gen_sample(f"{form_id}_{form_tile_id}", tile, [], original_resolution)
                continue

            num_words = random.randint(min_words_per_form, max_words_per_form) \
                if min_words_per_form != max_words_per_form \
                else min_words_per_form

            cur_words = all_words[cur_word: cur_word + num_words]
            cur_word += num_words
            yield _gen_sample(f"{form_id}_{form_tile_id}", tile, cur_words, original_resolution)


def save_dataset(out_dir: Path, dataset: Iterable[Sample], relative: bool = True):
    """
    Saves OCR dataset
    :param out_dir: directory where all images will be saved. It also puts gt.json file with
       ground truth description
    :param dataset: dataset to be saved
    """
    os.makedirs(out_dir, exist_ok=True)
    stored_dataset = _StoredDataset()
    for sample in dataset:
        id_norm = sample.sample_id.replace(".", "_")
        ext = ".png" if isinstance(sample.img, np.ndarray) else sample.img.suffix
        rel_path = Path(id_norm).with_suffix(ext)
        dest = out_dir / rel_path
        if isinstance(sample.img, np.ndarray):
            cv2.imwrite(str(dest), sample.img)
        else:
            shutil.copy(sample.img, dest)

        stored_dataset.samples[sample.sample_id] = sample.as_stored(rel_path if relative else dest.absolute())

    with open(out_dir / GROUND_TRUTH_FILENAME, "w") as f:
        f.write(stored_dataset.json())


def _load_xml(
    xml_root: Path,
    img_path_cb: Callable[[Document, Path], Path],
    lines_cb: Callable[[Document, str], List[Line]],
    no_load: bool,
) -> Iterable[Sample]:
    xmls = get_files(xml_root, extensions=".xml")
    for xml_path in xmls:
        dom: Document = xml.dom.minidom.parse(str(xml_path))
        form_id = xml_path.stem
        img_path = img_path_cb(dom, xml_path)

        if not img_path or not img_path.exists() or not img_path.stat().st_size:
            continue

        if no_load:
            img = img_path
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

        lines = lines_cb(dom, form_id)
        sample = Sample(sample_id=form_id, markup=lines, img=img)
        yield sample


def load_iam(
    iam_dir: Path,
    no_load: bool = False,
    valid_chars: Set[str] = None,
    words_ids_whitelist: Set[str] = None,
    words_ids_blacklist: Set[str] = None,
) -> Iterable[Sample]:
    """
    Loads IAM dataset
    :param iam_dir: root directory of IAM database
    :param no_load: don't load image, just put path into Sample.img field
    :param valid_chars: list of valid characters. If provided, all
        words which contains any other characters will be skipped
    :param words_ids_whitelist: list of whitelist word IDs.
        If provided all words that are not in whitelist will be ignored
    :param words_ids_blacklist: list of blacklisted word IDs.
        If provided any all words from blacklist will be skipped
    :return: collection of OCR dataset samples
    """
    xml_dir = iam_dir / "xml"

    def _img_path_cb(dom, xml_path: Path):
        return resolve_path(
            Path(xml_path.with_suffix(".png").name),
            xml_path.parent.parent / "forms"
        )

    def _get_word(node: Element):
        return Word(
            word_id=node.getAttribute("id"),
            text=node.getAttribute("text"),
            glyphs=[
                Rect.from_xywh(
                    int(rect_node.getAttribute("x")),
                    int(rect_node.getAttribute("y")),
                    int(rect_node.getAttribute("width")),
                    int(rect_node.getAttribute("height")),
                )
                for rect_node in node.getElementsByTagName("cmp")
            ]
        )

    def _is_word_valid(w: Word):
        if words_ids_whitelist and w.word_id not in words_ids_whitelist:
            return False

        if words_ids_blacklist and w.word_id in words_ids_blacklist:
            return False

        if valid_chars and set(w.text).difference(valid_chars):
            return False

        if not w.is_roi_valid():
            return False

        return True

    def _get_line(line_node: Element) -> Line:
        words = [
            word
            for word in map(_get_word, line_node.getElementsByTagName("word"))
            if _is_word_valid(word)
        ]
        text = " ".join(map(lambda w: w.text, words))
        return Line(text=text, words=words)

    def _get_lines(dom: Document, form_id: str) -> List[Line]:
        lines_dom = [
            _get_line(node)
            for node in dom.getElementsByTagName("line")
        ]
        valid_lines = map(lambda ln: bool(ln.words), lines_dom)
        return list(itertools.compress(lines_dom, valid_lines))

    return _load_xml(xml_dir, _img_path_cb, _get_lines, no_load)


def load_dataset(
    ds_dir: Path,
    shuffle: bool = True,
    no_load: bool = False,
    limit: int = None,
    progress_desc: str = None,
) -> Sequence[Sample]:
    """
    Loads dataset from ocr's module internal JSON format
    :param ds_dir: dataset directory should contain "gt.json"
    :param shuffle: whether to shuffle it after load
    :param no_load: whether to load images or just keep paths
    :param limit: limits amount of loaded items if present.
    :param progress_desc: if provided, then tqdm will be used to display progress bar.
    :return: list of dataset samples
    """

    gt_file = ds_dir / GROUND_TRUTH_FILENAME

    if not gt_file.exists():
        gt_file_old = ds_dir / GROUND_TRUTH_FILENAME_OLD
        if gt_file_old.exists():
            LOG.warning(f"Found ground truth in old format: {gt_file_old}")
            with open(gt_file_old) as f:
                ds_dict = json.load(f)
            ds = _StoredDataset(samples=from_jsonable(ds_dict))

            LOG.warning(f"    dataset loaded successfully from old-formatted file")

            # Also create GT in new format:
            with open(gt_file, "w") as f_new:
                f_new.write(ds.json(indent=4))
            LOG.warning(f"    generated {gt_file}")

    else:
        ds = _StoredDataset.parse_file(gt_file)

    samples = list(ds.samples.values())
    if shuffle:
        random.shuffle(samples)
    samples = samples[:limit]

    if not no_load:
        gtit = tqdm(samples, desc=progress_desc) if progress_desc else samples
        for sample in gtit:
            assert isinstance(sample.img, (Path, str))
            img_path = resolve_path(sample.img, gt_file.parent)
            assert img_path is not None
            sample.img = cv2.imread(str(img_path))

    return samples


def tf_load_dataset(
    ds_dir: Path,
    make_out_cb: Callable[[Sample], Any],
    batch_size: int,
    target_image_size: int,
    shuffle: bool = True,
):
    """
    Loads OCR dataset in tensorflow format, it uses TF methods only so it might
    be compatible with tensorflow graph compilation requirements
    :param ds_dir: dataset directory, must store gt.json file
    :param make_out_cb: callback to convert sample into GT output. Must be
        compatible with tensorflow types
    :param batch_size: size of dataset batch
    :param target_image_size: width and height of input image after resize
        of a single tile. It's recommended to set it to power of 2.
    :param shuffle: should dataset be shuffled?
    :return: dataset tensor in tensorflow format.
    """

    # Tensorflow runs heavy initialization, so keep its import
    # to be local whenever it is possible.
    import tensorflow as tf
    from tensorflow.python.data import AUTOTUNE

    gt = load_dataset(ds_dir, no_load=True)

    img_paths = []
    out_values = []
    for sample in gt:
        assert isinstance(sample.img, (Path, str))
        img_path = resolve_path(sample.img, ds_dir)
        img_paths.append(str(img_path))
        out_values.append(make_out_cb(sample))

    def _tf_input_output_callback(img_path: str, output):
        img_bytes = tf.io.read_file(img_path)
        image = tf.image.decode_image(img_bytes, expand_animations=False)
        image = tf.image.resize(image, (target_image_size, target_image_size))
        image = tf.cast(image, tf.float32) / 255.0 * 2. - 1.
        return image, output

    tf_ds = tf.data.Dataset.from_tensor_slices(
        (img_paths, out_values)
    ).map(_tf_input_output_callback, num_parallel_calls=AUTOTUNE)

    if shuffle:
        return tf_ds.shuffle(len(gt)).batch(batch_size).prefetch(AUTOTUNE).cache()
    return tf_ds.batch(batch_size).prefetch(AUTOTUNE).cache()


def load_page_xml(root_dir: Path, no_load: bool = False) -> Iterable[Sample]:
    """
    Loads PAGE-XML dataset
    :param root_dir: root directory of IAM database
    :param no_load: don't load image, just put path into Sample.img field
    :return: collection of OCR dataset samples.
    """

    def _get_page_dom(dom: Document) -> Element:
        pc_gts_node = dom.getElementsByTagName("PcGts").item(0)
        page = pc_gts_node.getElementsByTagName("Page").item(0)
        return page

    def _img_path_cb(dom: Document, xml_path: Path):
        page_dom = _get_page_dom(dom)
        img_path = page_dom.getAttribute("imageFilename")
        return resolve_path(img_path, root_dir)

    def _get_page_xml_word_text(w_node: Element):
        text_equiv = w_node.getElementsByTagName("TextEquiv").item(0)
        unicode_node = text_equiv.getElementsByTagName("Unicode").item(0)
        return unicode_node.firstChild.data

    def _get_page_xml_word_id(form_name, line_idx, word_idx):
        return f"page-xml-{form_name}-line_{line_idx}-word_{word_idx}"

    def _get_lines(dom: Document, form_id: str):
        page_dom = _get_page_dom(dom)
        words = page_dom.getElementsByTagName("Word")

        lines = []
        cur_line = []

        def _commit_cur_line():
            if not cur_line:
                return
            lines.append(Line(
                text=" ".join(map(lambda lnw: lnw.text, cur_line)),
                words=cur_line
            ))

        prev_word_left = -1
        prev_word_right = -1


        for w_dom in words:
            coords_node = w_dom.getElementsByTagName("Coords").item(0)
            points = coords_node.getAttribute("points").strip().split(" ")
            lt, rt, rb, lb = tuple([
                tuple(map(int, pt.split(",")))
                for pt in points
            ])
            left, top = lt
            right, bottom = rb

            def _mk_word(word_idx):
                return Word(
                    word_id=_get_page_xml_word_id(form_id, len(lines), word_idx),
                    text=_get_page_xml_word_text(w_dom),
                    glyphs=[Rect.from_xywh(left, top, right-left, bottom-top)]
                )

            is_new_line = (
                prev_word_left <= left <= prev_word_right
                or prev_word_left <= right <= prev_word_right
            )
            word_idx = 0 if is_new_line else len(cur_line)
            w = _mk_word(word_idx)
            if not w.is_roi_valid():
                continue

            if not is_new_line:
                cur_line.append(_mk_word(len(cur_line)))
            else:
                assert cur_line, (
                    "We expect at least one word to be in cur line"
                    " before new-line is triggered."
                )
                _commit_cur_line()
                cur_line = [_mk_word(0)]
            prev_word_left, _ = lt
            prev_word_right, _ = rb

        _commit_cur_line()
        return lines

    return _load_xml(root_dir, _img_path_cb, _get_lines, no_load)
