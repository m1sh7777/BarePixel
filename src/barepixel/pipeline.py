"""Sequential, reusable image processing operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .filters import gaussian_blur, sobel
from .normalize import normalize
from .tensor import ImageTensor, as_image_array

Operation = Callable[[NDArray[np.number]], NDArray[np.number]]


@dataclass(frozen=True)
class GaussianBlur:
    """Reusable Gaussian blur operation."""

    kernel_size: int = 5
    sigma: float = 1.0

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.float64]:
        """Blur an image with the configured kernel."""
        return gaussian_blur(image, self.kernel_size, self.sigma)


@dataclass(frozen=True)
class Sobel:
    """Reusable Sobel gradient operation."""

    axis: str = "both"

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.float64]:
        """Return the configured image gradient."""
        return sobel(image, self.axis)


@dataclass(frozen=True)
class Normalize:
    """Reusable global range scaling operation."""

    min_value: float = 0.0
    max_value: float = 1.0

    def __call__(self, image: NDArray[np.number]) -> NDArray[np.float64]:
        """Scale an image to the configured range."""
        return normalize(image, self.min_value, self.max_value)


class Pipeline:
    """Execute image operations in sequence without intermediate copies."""

    def __init__(self, operations: Iterable[Operation]) -> None:
        """Store callable operations; raise TypeError for non-callables."""
        self.operations = tuple(operations)
        if any(not callable(operation) for operation in self.operations):
            raise TypeError("every pipeline operation must be callable")

    def run(self, image: ImageTensor | ArrayLike) -> NDArray[np.number]:
        """Return the final image; propagate operation errors with step context."""
        result = as_image_array(image)
        for index, operation in enumerate(self.operations):
            try:
                result = as_image_array(operation(result))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"pipeline step {index} ({type(operation).__name__}): {error}"
                ) from error
        return result
