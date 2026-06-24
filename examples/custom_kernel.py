"""Apply a custom sharpening kernel to generated data."""

from pathlib import Path

import numpy as np

import barepixel as vc


def main():
    output = Path(__file__).parent / "output"
    output.mkdir(exist_ok=True)
    image = np.zeros((128, 128), dtype=np.uint8)
    image[24:104, 24:104] = 150
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = vc.convolve2d(image, kernel)
    vc.imwrite(output / "sharpened.png", np.clip(sharpened, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    main()
