"""Blur and normalize a generated grayscale image."""

from pathlib import Path

import numpy as np

import barepixel as vc


def main():
    output = Path(__file__).parent / "output"
    output.mkdir(exist_ok=True)
    image = np.tile(np.arange(256, dtype=np.uint8), (128, 1))
    vc.imwrite(output / "basic_input.png", image)
    result = vc.normalize(
        vc.gaussian_blur(vc.imread(output / "basic_input.png", grayscale=True))
    )
    vc.imwrite(output / "basic_result.png", np.rint(result * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
