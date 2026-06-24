"""Validated NumPy image storage."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .exceptions import ImageValidationError


def as_image_array(image: ImageTensor | ArrayLike) -> NDArray[np.number]:
    """Return a validated image view; raise ImageValidationError for invalid images."""
    array = image._data if isinstance(image, ImageTensor) else np.asarray(image)
    if array.ndim not in (2, 3) or any(size == 0 for size in array.shape):
        raise ImageValidationError("image must have nonempty (H, W) or (H, W, C) shape")
    if array.dtype.kind not in "iuf":
        raise ImageValidationError("image dtype must be real integer or floating point")
    return array


class ImageTensor:
    """Wrap a 2D or 3D numeric array; copy=False shares storage by default."""

    def __init__(self, data: ArrayLike, *, copy: bool = False) -> None:
        """Validate data and optionally copy it.

        Invalid shapes or dtypes raise ImageValidationError.
        """
        self._data = as_image_array(
            np.array(data, copy=True) if copy else np.asarray(data)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the image shape."""
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the image data type."""
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self._data.ndim

    @property
    def channels(self) -> int:
        """Return one for grayscale, otherwise the final-axis size."""
        return 1 if self.ndim == 2 else self.shape[2]

    def numpy(self, *, copy: bool = False) -> NDArray[np.number]:
        """Return a shared NumPy view, or an independent array when copy=True."""
        return self._data.copy() if copy else self._data

    def copy(self) -> ImageTensor:
        """Return an independent tensor with copied data."""
        return ImageTensor(self._data, copy=True)

    def astype(self, dtype: DTypeLike) -> ImageTensor:
        """Return an independent tensor converted to a real numeric dtype."""
        target = np.dtype(dtype)
        if target.kind not in "iuf":
            raise ImageValidationError("target dtype must be real numeric")
        return ImageTensor(self._data.astype(target, copy=True))
