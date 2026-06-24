import numpy as np
import pytest

from barepixel import normalize


def test_range_and_constant():
    np.testing.assert_array_equal(normalize(np.array([[2, 4]]), -1, 1), [[-1, 1]])
    np.testing.assert_array_equal(
        normalize(np.ones((2, 2, 3)), 5, 9), np.full((2, 2, 3), 5)
    )


@pytest.mark.parametrize("low,high", [(1, 1), (2, 1), (0, np.inf)])
def test_bad_range(low, high):
    with pytest.raises(ValueError):
        normalize(np.ones((2, 2)), low, high)
