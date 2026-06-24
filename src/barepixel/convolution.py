"""Mathematical convolution with zero padding."""

from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from .tensor import ImageTensor, as_image_array


def _prepare(
    image: ImageTensor | ArrayLike, kernel: ArrayLike, stride: int, padding: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int]:
    data = as_image_array(image).astype(np.float64)
    weights = np.asarray(kernel)
    if weights.ndim != 2 or 0 in weights.shape or weights.dtype.kind not in "iuf":
        raise ValueError("kernel must be a nonempty real numeric 2D array")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if padding not in ("same", "valid"):
        raise ValueError("padding must be 'same' or 'valid'")
    height, width = data.shape[:2]
    kh, kw = weights.shape
    if padding == "valid":
        if kh > height or kw > width:
            raise ValueError("valid padding requires kernel no larger than image")
        top = left = bottom = right = 0
    else:
        output_h, output_w = math.ceil(height / stride), math.ceil(width / stride)
        pad_h = max((output_h - 1) * stride + kh - height, 0)
        pad_w = max((output_w - 1) * stride + kw - width, 0)
        top, left = pad_h // 2, pad_w // 2
        bottom, right = pad_h - top, pad_w - left
    pads = ((top, bottom), (left, right))
    if data.ndim == 3:
        pads += ((0, 0),)
    return np.pad(data, pads), weights.astype(np.float64)[::-1, ::-1], top, left


def convolve2d(
    image: ImageTensor | ArrayLike,
    kernel: ArrayLike,
    stride: int = 1,
    padding: str = "same",
) -> NDArray[np.float64]:
    """Convolve each channel in float64 with zero padding and return a NumPy array.

    For same padding, output spatial size is ceil(input/stride); extra padding goes
    to the bottom and right. Valid padding uses no border and rejects oversized kernels.
    """
    data, weights, _, _ = _prepare(image, kernel, stride, padding)
    kh, kw = weights.shape
    windows = sliding_window_view(data, (kh, kw), axis=(0, 1))[::stride, ::stride]
    if data.ndim == 2:
        return np.einsum("ijxy,xy->ij", windows, weights, optimize=True)
    return np.einsum("ijcxy,xy->ijc", windows, weights, optimize=True)


def convolve2d_reference(
    image: ImageTensor | ArrayLike,
    kernel: ArrayLike,
    stride: int = 1,
    padding: str = "same",
) -> NDArray[np.float64]:
    """Convolve with explicit Python loops for learning and benchmarking."""
    data, weights, _, _ = _prepare(image, kernel, stride, padding)
    kh, kw = weights.shape
    out_h = (data.shape[0] - kh) // stride + 1
    out_w = (data.shape[1] - kw) // stride + 1
    result = np.empty((out_h, out_w, *data.shape[2:]), dtype=np.float64)
    for y in range(out_h):
        for x in range(out_w):
            if data.ndim == 2:
                total = 0.0
                for ky in range(kh):
                    for kx in range(kw):
                        total += float(data[y * stride + ky, x * stride + kx]) * float(
                            weights[ky, kx]
                        )
                result[y, x] = total
            else:
                for channel in range(data.shape[2]):
                    total = 0.0
                    for ky in range(kh):
                        for kx in range(kw):
                            total += float(
                                data[y * stride + ky, x * stride + kx, channel]
                            ) * float(weights[ky, kx])
                    result[y, x, channel] = total
    return result
