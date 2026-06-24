import cv2
import numpy as np
import pytest

from barepixel import dilate, erode


@pytest.mark.parametrize("size,iterations", [((3, 3), 1), ((2, 4), 2)])
def test_against_opencv(size, iterations):
    image = np.random.default_rng(3).integers(0, 256, size=(7, 8), dtype=np.uint8)
    kernel = np.ones(size, dtype=np.uint8)
    np.testing.assert_array_equal(
        erode(image, size, iterations),
        cv2.erode(image, kernel, iterations=iterations, borderType=cv2.BORDER_CONSTANT),
    )
    np.testing.assert_array_equal(
        dilate(image, size, iterations),
        cv2.dilate(
            image, kernel, iterations=iterations, borderType=cv2.BORDER_CONSTANT
        ),
    )


@pytest.mark.parametrize("size,iterations", [(0, 1), ((2, 0), 1), (3, 0)])
def test_invalid_morphology(size, iterations):
    with pytest.raises(ValueError):
        erode(np.ones((3, 3)), size, iterations)


def test_large_integer_values_keep_precision():
    values = np.array([[2**53 + 1, 2**53 + 2]], dtype=np.uint64)
    np.testing.assert_array_equal(erode(values, (1, 2)), [[2**53 + 1, 2**53 + 1]])
    np.testing.assert_array_equal(dilate(values, (1, 2)), [[2**53 + 1, 2**53 + 2]])
