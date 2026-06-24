"""Compare equivalent zero-padded mathematical convolutions."""

from __future__ import annotations

import statistics
import time

import cv2
import numpy as np

from barepixel import convolve2d, convolve2d_reference


def measure(function, image, kernel, repeats=5):
    function(image, kernel)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(image, kernel)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def opencv_reference(image, kernel):
    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh - 1 - kh // 2), (kw // 2, kw - 1 - kw // 2)))
    filtered = cv2.filter2D(
        padded,
        cv2.CV_64F,
        kernel[::-1, ::-1],
        anchor=(0, 0),
        borderType=cv2.BORDER_CONSTANT,
    )
    return filtered[: image.shape[0], : image.shape[1]]


def main():
    rng = np.random.default_rng(42)
    for size in (32, 64, 128):
        image = rng.normal(size=(size, size))
        for width in (3, 5):
            kernel = rng.normal(size=(width, width))
            expected = convolve2d(image, kernel)
            np.testing.assert_allclose(
                opencv_reference(image, kernel), expected, atol=1e-10
            )
            print(f"image={size}x{size}, kernel={width}x{width}")
            for name, function in (
                ("Python loops", convolve2d_reference),
                ("NumPy", convolve2d),
                ("OpenCV", opencv_reference),
            ):
                print(f"  {name}: {measure(function, image, kernel):.6f} s median")


if __name__ == "__main__":
    main()
