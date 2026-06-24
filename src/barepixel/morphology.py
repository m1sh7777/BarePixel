"""Grayscale rectangular morphology."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from .tensor import ImageTensor, as_image_array


def _morph(
    image: ImageTensor | ArrayLike,
    kernel_size: int | tuple[int, int],
    iterations: int,
    *,
    erosion: bool,
) -> NDArray[np.number]:
    data = as_image_array(image)
    if data.ndim != 2:
        raise ValueError("morphology requires a grayscale image")
    if isinstance(kernel_size, int) and not isinstance(kernel_size, bool):
        kernel_size = (kernel_size, kernel_size)
    if (
        not isinstance(kernel_size, tuple)
        or len(kernel_size) != 2
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in kernel_size
        )
    ):
        raise ValueError("kernel_size must be a positive integer or (height, width)")
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or iterations < 1
    ):
        raise ValueError("iterations must be a positive integer")
    kh, kw = kernel_size
    result = data.copy()
    if data.dtype.kind == "f":
        fill = np.inf if erosion else -np.inf
    else:
        bounds = np.iinfo(data.dtype)
        fill = bounds.max if erosion else bounds.min
    for _ in range(iterations):
        padded = np.pad(
            result,
            ((kh // 2, kh - 1 - kh // 2), (kw // 2, kw - 1 - kw // 2)),
            constant_values=fill,
        )
        windows = sliding_window_view(padded, (kh, kw))
        result = windows.min(axis=(-2, -1)) if erosion else windows.max(axis=(-2, -1))
    return result


def erode(
    image: ImageTensor | ArrayLike,
    kernel_size: int | tuple[int, int] = 3,
    iterations: int = 1,
) -> NDArray[np.number]:
    """Apply grayscale erosion; outside pixels have neutral +infinity value."""
    return _morph(image, kernel_size, iterations, erosion=True)


def dilate(
    image: ImageTensor | ArrayLike,
    kernel_size: int | tuple[int, int] = 3,
    iterations: int = 1,
) -> NDArray[np.number]:
    """Apply grayscale dilation; outside pixels have neutral -infinity value."""
    return _morph(image, kernel_size, iterations, erosion=False)
