"""BarePixel exceptions."""


class BarePixelError(Exception):
    """Base error for BarePixel operations."""


class ImageValidationError(BarePixelError, ValueError):
    """Raised when image shape or dtype is unsupported."""
