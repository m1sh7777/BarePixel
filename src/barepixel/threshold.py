"""Binary image thresholding."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .tensor import ImageTensor, as_image_array


def threshold(
    image: ImageTensor | ArrayLike,
    threshold: float = 128,
    max_value: float = 255,
    *,
    inverse: bool = False,
) -> NDArray[np.float64]:
    """Return 0 or max_value; equality is below threshold, including in inverse mode."""
    data = as_image_array(image)
    if data.ndim != 2:
        raise ValueError("threshold requires a grayscale image")
    if not np.isfinite(threshold) or not np.isfinite(max_value) or max_value <= 0:
        raise ValueError("threshold must be finite and max_value positive and finite")
    if not isinstance(inverse, bool):
        raise ValueError("inverse must be a bool")
    selected = data <= threshold if inverse else data > threshold
    return np.where(selected, max_value, 0.0).astype(np.float64)
