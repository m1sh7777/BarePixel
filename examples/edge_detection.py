"""Find edges in a generated image."""

from pathlib import Path

import numpy as np

import barepixel as vc


def main():
    output = Path(__file__).parent / "output"
    output.mkdir(exist_ok=True)
    image = np.zeros((128, 128), dtype=np.uint8)
    image[32:96, 32:96] = 200
    result = vc.Pipeline([vc.GaussianBlur(5), vc.Sobel(), vc.Normalize()]).run(image)
    vc.imwrite(output / "edges.png", np.rint(result * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
