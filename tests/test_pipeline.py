import numpy as np
import pytest

from barepixel import GaussianBlur, Normalize, Pipeline, Sobel


def test_pipeline_reuse_and_error_context():
    image = np.arange(25).reshape(5, 5)
    pipeline = Pipeline([GaussianBlur(3), Sobel(), Normalize()])
    result = pipeline.run(image)
    np.testing.assert_array_equal(result, pipeline.run(image))
    assert result.min() == 0 and result.max() == 1
    with pytest.raises(ValueError, match="pipeline step 0"):
        Pipeline([Sobel(axis="bad")]).run(image)
    with pytest.raises(TypeError):
        Pipeline([42])
