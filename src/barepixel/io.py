"""OpenCV-backed image files using BGR channel order."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import ArrayLike

from .tensor import ImageTensor, as_image_array


def imread(path: str | Path, *, grayscale: bool = False) -> ImageTensor:
    """Load an image as uint8 BGR or grayscale; raise on missing or unreadable files."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(source), flag)
    if image is None:
        raise ValueError(f"cannot read image: {source}")
    return ImageTensor(image)


def imwrite(path: str | Path, image: ImageTensor | ArrayLike) -> None:
    """Write a uint8 BGR or grayscale PNG/JPEG; reject unsupported types and paths."""
    target = Path(path)
    if target.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("output must be PNG or JPEG")
    data = as_image_array(image)
    if data.dtype != np.uint8 or (data.ndim == 3 and data.shape[2] != 3):
        raise ValueError("output must be uint8 grayscale or three-channel BGR")
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    if not cv2.imwrite(str(target), data):
        raise OSError(f"cannot write image: {target}")
