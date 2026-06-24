"""Numerical image processing with NumPy."""

from .convolution import convolve2d, convolve2d_reference
from .exceptions import BarePixelError, ImageValidationError
from .filters import gaussian_blur, gaussian_kernel, sobel
from .io import imread, imwrite
from .morphology import dilate, erode
from .normalize import normalize
from .pipeline import GaussianBlur, Normalize, Pipeline, Sobel
from .tensor import ImageTensor
from .threshold import threshold

__version__ = "0.2.0"

__all__ = [
    "GaussianBlur",
    "ImageTensor",
    "ImageValidationError",
    "Normalize",
    "Pipeline",
    "Sobel",
    "BarePixelError",
    "convolve2d",
    "convolve2d_reference",
    "dilate",
    "erode",
    "gaussian_blur",
    "gaussian_kernel",
    "imread",
    "imwrite",
    "normalize",
    "sobel",
    "threshold",
]
