import numpy as np
import pytest

from barepixel import ImageTensor, ImageValidationError


def test_storage_and_copy_semantics():
    data = np.zeros((2, 3, 3), dtype=np.uint8)
    tensor = ImageTensor(data)
    assert (tensor.shape, tensor.ndim, tensor.channels, tensor.dtype) == (
        (2, 3, 3),
        3,
        3,
        np.uint8,
    )
    data[0, 0, 0] = 7
    assert tensor.numpy()[0, 0, 0] == 7
    tensor.copy().numpy()[0, 0, 0] = 9
    assert data[0, 0, 0] == 7
    assert tensor.astype(np.float32).dtype == np.float32
    assert ImageTensor(np.ones((2, 2))).channels == 1


@pytest.mark.parametrize(
    "data",
    [np.ones(3), np.ones((1, 1, 1, 1)), np.ones((0, 2)), np.ones((2, 2), dtype=bool)],
)
def test_rejects_invalid_images(data):
    with pytest.raises(ImageValidationError):
        ImageTensor(data)
