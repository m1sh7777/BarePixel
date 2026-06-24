"""Generate a comparison figure from a synthetic grayscale image."""

from pathlib import Path

import cv2
import numpy as np

import barepixel as cv


def synthetic_image() -> np.ndarray:
    """Return a deterministic grayscale scene made from basic array shapes."""
    image = np.tile(np.linspace(25, 80, 360, dtype=np.uint8), (180, 1))
    image[34:146, 38:142] = 175
    image[20:160, 175:179] = 245
    y, x = np.ogrid[:180, :360]
    image[(x - 265) ** 2 + (y - 90) ** 2 <= 47**2] = 225
    image[74:106, 251:279] = 55
    return image


def labeled_panel(label: str, image: np.ndarray) -> np.ndarray:
    """Add a plain label above a grayscale result without changing its pixels."""
    header = np.full((36, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        label,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    color = np.repeat(image[:, :, None], 3, axis=2)
    return np.vstack((header, color))


def main() -> None:
    """Write assets/examples.png using BarePixel operations."""
    original = synthetic_image()
    blurred = np.rint(cv.gaussian_blur(original, 7, 1.5)).astype(np.uint8)
    magnitude = cv.sobel(original)
    edges = np.rint(cv.normalize(magnitude, 0, 255)).astype(np.uint8)
    binary = cv.threshold(original, threshold=128).astype(np.uint8)
    eroded = cv.erode(binary, kernel_size=7)
    dilated = cv.dilate(binary, kernel_size=7)
    results = (
        ("Original", original),
        ("Gaussian blur", blurred),
        ("Sobel magnitude (display scaled)", edges),
        ("Binary threshold", binary),
        ("Erosion of binary mask", eroded),
        ("Dilation of binary mask", dilated),
    )
    comparison = np.vstack([labeled_panel(label, image) for label, image in results])
    target = Path(__file__).resolve().parents[1] / "assets" / "examples.png"
    target.parent.mkdir(exist_ok=True)
    cv.imwrite(target, comparison)


if __name__ == "__main__":
    main()
