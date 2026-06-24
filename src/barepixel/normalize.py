"""Image range scaling."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .tensor import ImageTensor, as_image_array


def normalize(
    image: ImageTensor | ArrayLike, min_value: float = 0.0, max_value: float = 1.0
) -> NDArray[np.float64]:
    """Scale globally to [min_value, max_value]; constant images map to min_value."""
    if (
        not np.isfinite(min_value)
        or not np.isfinite(max_value)
        or min_value >= max_value
    ):
        raise ValueError("output range must be finite and increasing")
    data = as_image_array(image).astype(np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("image must contain only finite values")
    low, high = data.min(), data.max()
    if low == high:
        return np.full(data.shape, min_value, dtype=np.float64)
    return (data - low) * ((max_value - min_value) / (high - low)) + min_value
