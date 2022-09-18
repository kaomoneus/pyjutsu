import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence, Iterable

from pyjutsu.dataset.iam_lists.blacklist import IAM_BLACKLIST
from pyjutsu.dataset.iam_lists.whitelist import IAM_WHITELIST
from pyjutsu.dataset.ocr import load_iam, _StoredDataset, load_page_xml, load_dataset, save_dataset, Sample
from test_pyjutsu.config import RESOURCES_BASE


def test_load_iam_simple():

    iam_root = RESOURCES_BASE / "iam_ds"

    res = load_iam(iam_dir=iam_root)

    dataset = _StoredDataset.from_samples(res)

    assert len(dataset.samples) == 2
    assert len(dataset.samples["form_iam_0"].markup) == 2
    assert len(dataset.samples["form_iam_0"].markup[1].words) == 2
    assert dataset.samples["form_iam_0"].markup[0].words[1].text == "world"
    assert len(dataset.samples["form_iam_1"].markup) == 2


def test_load_iam_validity_lists():

    iam_root = RESOURCES_BASE / "iam_ds"

    alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    all_words = {
        "form_iam_0-0",
        "form_iam_0-1",
        "form_iam_0-2",
        "form_iam_0-3",
        "form_iam_1-0",
        "form_iam_1-1",
        "form_iam_1-2",
    }

    valid_words = all_words.difference({"form_iam_0-3"})
    invalid_words = {"form_iam_0-1"}

    res = load_iam(
        iam_dir=iam_root,
        no_load=False,
        valid_chars=alphabet.difference({"H"}),
        words_ids_whitelist=valid_words,
        words_ids_blacklist=invalid_words
    )

    dataset = _StoredDataset.from_samples(res)

    assert len(dataset.samples) == 2
    assert len(dataset.samples["form_iam_0"].markup) == 1
    assert len(dataset.samples["form_iam_0"].markup[0].words) == 1
    assert dataset.samples["form_iam_0"].markup[0].words[0].text == "Second"
    assert len(dataset.samples["form_iam_1"].markup) == 2


def test_load_page_xml_simple():

    xml_root = RESOURCES_BASE / "page_xml_ds"

    res = load_page_xml(root_dir=xml_root)

    dataset = _StoredDataset.from_samples(res)

    assert len(dataset.samples) == 3
    assert len(dataset.samples["form_page_xml_0"].markup) == 2
    assert len(dataset.samples["form_page_xml_0"].markup[0].words) == 2
    assert len(dataset.samples["form_page_xml_1"].markup[1].words) == 1

    assert len(dataset.samples["form_page_xml_2"].markup) == 1
    assert len(dataset.samples["form_page_xml_2"].markup[0].words) == 2
    assert dataset.samples["form_page_xml_2"].markup[0].words[0].text == "Hello"


def _sorted_samples(samples: Iterable[Sample]):
    return sorted(samples, key=lambda s: s.sample_id)


def test_save_load_dataset():
    xml_root = RESOURCES_BASE / "page_xml_ds"
    samples_ds = _sorted_samples(load_page_xml(root_dir=xml_root))

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ds_path = tmp / "ds"

        save_dataset(ds_path, samples_ds)

        loaded_ds = _StoredDataset.from_samples(
            load_dataset(ds_path, shuffle=False)
        )

        assert samples_ds == _sorted_samples(loaded_ds.samples.values())

    samples_ds_noload = _sorted_samples(load_page_xml(
        root_dir=xml_root,
        no_load=True
    ))

    assert isinstance(samples_ds_noload[0].img, Path)


def test_iam_lists_non_empty():
    # Update these numbers each time you
    # do manual check session.
    assert len(IAM_WHITELIST) >= 21795
    assert len(IAM_BLACKLIST) >= 3627
