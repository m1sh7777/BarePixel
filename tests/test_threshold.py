import numpy as np
import pytest

from barepixel import threshold


def test_binary_boundary_and_inverse():
    image = np.array([[127, 128, 129]])
    np.testing.assert_array_equal(threshold(image), [[0, 0, 255]])
    np.testing.assert_array_equal(threshold(image, inverse=True), [[255, 255, 0]])


def test_invalid_threshold():
    with pytest.raises(ValueError):
        threshold(np.ones((2, 2, 3)))
    with pytest.raises(ValueError):
        threshold(np.ones((2, 2)), max_value=0)
