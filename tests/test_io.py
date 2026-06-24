import numpy as np
import pytest

from barepixel import imread, imwrite


def test_png_round_trip(tmp_path):
    image = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    path = tmp_path / "color.png"
    imwrite(path, image)
    np.testing.assert_array_equal(imread(path).numpy(), image)
    assert imread(path, grayscale=True).ndim == 2


def test_io_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        imread(tmp_path / "missing.png")
    with pytest.raises(ValueError):
        imwrite(tmp_path / "bad.png", np.ones((2, 2), dtype=np.float64))
