import cv2
import numpy as np
import pytest

from barepixel import gaussian_blur, gaussian_kernel, sobel


def test_gaussian_kernel_and_opencv_compatible_interior():
    image = np.arange(49, dtype=np.float64).reshape(7, 7)
    kernel = gaussian_kernel(3, 1.2)
    assert kernel.sum() == pytest.approx(1)
    result = gaussian_blur(image, 3, 1.2)
    expected = cv2.GaussianBlur(image, (3, 3), 1.2, borderType=cv2.BORDER_CONSTANT)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    assert gaussian_blur(np.ones((3, 3, 3))).shape == (3, 3, 3)


def test_sobel_gradients_and_magnitude():
    image = np.tile(np.arange(5, dtype=np.uint8), (5, 1))
    assert sobel(image, "x")[2, 2] == -8
    assert sobel(image, "y")[2, 2] == 0
    assert sobel(image, "both")[2, 2] == 8
    color = np.stack((image, image, image), axis=-1)
    np.testing.assert_allclose(sobel(color)[2, 2], 8)


@pytest.mark.parametrize(
    "call",
    [
        lambda: gaussian_kernel(2),
        lambda: gaussian_kernel(3, 0),
        lambda: sobel(np.ones((3, 3)), "z"),
    ],
)
def test_invalid_filter_arguments(call):
    with pytest.raises(ValueError):
        call()
