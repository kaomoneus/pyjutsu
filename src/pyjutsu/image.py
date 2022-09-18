"""
This package contains dataset helpers for working with images.

Note, there are plenty of libs to work with images, this module
contains just very thin and very top level stuff, to work with them.
"""
import dataclasses
import itertools
import math
from enum import Enum, auto
from functools import cached_property
from typing import Dict

import cv2
import numpy as np
from pydantic import BaseModel

from pyjutsu.errors import Error
from scipy import interpolate


CM_PER_INCH = 2.54


class SheetDimensions(BaseModel):
    width_mm: int
    height_mm: int

    # TODO: cached property
    def _width_to_height(self):
        return self.width_mm / self.height_mm

    # TODO: cached property
    def _area_sq_mm(self):
        return self.width_mm * self.height_mm

    # TODO: cached property
    def _area_inches(self):
        return self._area_sq_mm() / (100. * CM_PER_INCH**2.)

    def get_dpi_for(self, img: np.ndarray, strict: bool = False):
        if isinstance(img, (np.ndarray, np.generic)):
            img_width_to_height = img.shape[1] / img.shape[0]
        # elif isinstance(img, scipy.ndimage):
        #     # TODO
        else:
            raise Error(f"Unknown image type: {type(img)}")

        if strict and math.fabs(self._width_to_height() - img_width_to_height) > 0.01:
            return None

        img_area = img.shape[0] * img.shape[1]

        return (img_area / self._area_inches())**0.5

    @staticmethod
    def get_a4():
        return SheetDimensions(width_mm=210, height_mm=297)


class SheetFormat(Enum):
    """
    Sheet format values given as pairs of
    width and height dimensions in millimiters,
    assuming it is in `protrait` orientation.
    """
    A4 = SheetDimensions(width_mm=210, height_mm=297)


def magnie_humie(src_img: np.ndarray):
    width = src_img.shape[1]
    height = src_img.shape[0]

    maxy = height-1

    linear_zone = 0.85
    pixmove = 0.04

    left_mover = 0.5 - linear_zone/2
    right_mover = 0.5 + linear_zone/2

    y_src = np.array([0., left_mover, 0.5, right_mover, 1]) * maxy
    y_magnie = np.array([0., left_mover+pixmove, 0.5, right_mover-pixmove, 1]) * maxy
    y_humie = np.array([0., left_mover-pixmove, 0.5, right_mover+pixmove, 1]) * maxy

    magnie_tck = interpolate.splrep(y_src, y_magnie)
    humie_tck = interpolate.splrep(y_src, y_humie)
    y_args = [*map(float, range(height))]

    magnie_map_y = interpolate.splev(y_args, magnie_tck)
    humie_map_y = interpolate.splev(y_args, humie_tck)

    magnie_map_y = np.array([magnie_map_y]*width, dtype=np.float32).transpose()
    humie_map_y = np.array([humie_map_y]*width, dtype=np.float32).transpose()

    map_x = np.array([[x for x in range(width)]] * height, np.float32)

    magnie = cv2.remap(src_img, map_x, magnie_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255.)
    humie = cv2.remap(src_img, map_x, humie_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255.)

    return magnie, humie


class AUGThresholdMode(Enum):
    NO_THRESHOLD = auto()
    ONLY_THRESHOLD = auto()
    FULL = auto()


def augment_image(
    src_img: np.ndarray,
    threshold_mode: AUGThresholdMode,
) -> Dict[str, np.ndarray]:
    """
    Augments image
    :param src_img: image to be augmented
    :param threshold_mode: thresholding mode
    :return: dictionary where key is augmentation name, value is augmented image
    """
    res = {}

    threshold_value, thr_mid = cv2.threshold(src_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    res["threshold"] = thr_mid
    if threshold_mode == AUGThresholdMode.ONLY_THRESHOLD:
        return res

    blurred = cv2.GaussianBlur(thr_mid, (3, 3), sigmaX=1.)

    thr_val_left = int(threshold_value * 0.95)
    thr_val_right = int(threshold_value * 1.2)

    _, thr_left = cv2.threshold(blurred, thr_val_left, 255, cv2.THRESH_BINARY)
    _, thr_right = cv2.threshold(blurred, thr_val_right, 255, cv2.THRESH_BINARY)
    adaptive_threshold = cv2.adaptiveThreshold(
        src_img,
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3
    )

    res.update({
        "blurred": blurred,
        "adaptive_threshold": adaptive_threshold,
        "threshold_left": thr_left,
        "threshold_right": thr_right,
    })

    return res


def get_tiles(src_img: np.ndarray, rows: int, cols: int):
    """
    Split source image onto set of tiles
    :param src_img: source image
    :param rows: amount of tiles among vertical axis
    :param cols: amount of tiles among horizontal axis
    :return: list of tiles
    """
    h, w = src_img.shape[:2]
    tiles = []
    for r, c in itertools.product(range(rows), range(cols)):
        r_start = r * h // rows
        r_end = (r+1) * h // rows
        c_start = c * w // cols
        c_end = (c+1) * w // cols
        tiles.append(src_img[r_start:r_end, c_start:c_end])

    return tiles
