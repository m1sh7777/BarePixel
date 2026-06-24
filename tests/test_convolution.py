import numpy as np
import pytest
from scipy.signal import convolve2d as scipy_convolve2d

from barepixel import convolve2d, convolve2d_reference


@pytest.mark.parametrize(
    "shape,kernel_shape,stride,padding",
    [
        ((5, 7), (3, 3), 1, "same"),
        ((5, 7), (2, 4), 2, "same"),
        ((6, 5), (4, 2), 2, "valid"),
        ((5, 7, 3), (2, 3), 1, "same"),
    ],
)
def test_vectorized_matches_reference(shape, kernel_shape, stride, padding):
    rng = np.random.default_rng(4)
    image = rng.normal(size=shape)
    kernel = rng.normal(size=kernel_shape)
    np.testing.assert_allclose(
        convolve2d(image, kernel, stride, padding),
        convolve2d_reference(image, kernel, stride, padding),
    )


def test_true_convolution_and_scipy_reference():
    image = np.arange(1, 10).reshape(3, 3)
    kernel = np.array([[1, 2], [3, 4]])
    actual = convolve2d(image, kernel, padding="valid")
    np.testing.assert_array_equal(actual, scipy_convolve2d(image, kernel, mode="valid"))
    assert actual[0, 0] == 23


def test_same_stride_shape_and_asymmetric_padding():
    result = convolve2d(np.ones((4, 5)), np.ones((2, 4)), stride=2)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(
        result, convolve2d_reference(np.ones((4, 5)), np.ones((2, 4)), stride=2)
    )


@pytest.mark.parametrize(
    "kernel,stride,padding",
    [
        (np.ones((5, 2)), 1, "valid"),
        (np.ones((2, 2)), 0, "same"),
        (np.ones((2, 2)), 1, "reflect"),
        (np.ones(3), 1, "same"),
    ],
)
def test_invalid_arguments(kernel, stride, padding):
    with pytest.raises(ValueError):
        convolve2d(np.ones((3, 3)), kernel, stride, padding)
