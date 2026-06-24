"""Image filters built on BarePixel convolution."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .convolution import convolve2d
from .tensor import ImageTensor, as_image_array


def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> NDArray[np.float64]:
    """Return a normalized odd, square Gaussian kernel; reject invalid parameters."""
    if (
        isinstance(kernel_size, bool)
        or not isinstance(kernel_size, int)
        or kernel_size < 1
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be a positive odd integer")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    coordinates = np.arange(kernel_size, dtype=np.float64) - kernel_size // 2
    x, y = np.meshgrid(coordinates, coordinates)
    weights = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return weights / weights.sum()


def gaussian_blur(
    image: ImageTensor | ArrayLike, kernel_size: int = 5, sigma: float = 1.0
) -> NDArray[np.float64]:
    """Blur with a Gaussian kernel and zero-padded BarePixel convolution."""
    return convolve2d(image, gaussian_kernel(kernel_size, sigma))


def sobel(image: ImageTensor | ArrayLike, axis: str = "both") -> NDArray[np.float64]:
    """Return x, y, or magnitude gradient as float64, with zero-padded borders.

    Three-channel BGR is converted using 0.114 B + 0.587 G + 0.299 R.
    """
    if axis not in ("x", "y", "both"):
        raise ValueError("axis must be 'x', 'y', or 'both'")
    data = as_image_array(image).astype(np.float64)
    if data.ndim == 3:
        if data.shape[2] != 3:
            raise ValueError("Sobel color input must have three BGR channels")
        data = np.einsum("hwc,c->hw", data, [0.114, 0.587, 0.299])
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = kernel_x.T
    gx = convolve2d(data, kernel_x) if axis in ("x", "both") else None
    gy = convolve2d(data, kernel_y) if axis in ("y", "both") else None
    if axis == "x":
        return gx
    if axis == "y":
        return gy
    return np.hypot(gx, gy)
