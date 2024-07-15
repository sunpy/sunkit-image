from pathlib import Path

import numpy as np
import pytest
import sunpy.data.test


@pytest.fixture()
def aia171_test_clipping():
    return np.asarray([0.2, -0.3, -1.0001])


@pytest.fixture()
def aia171_test_map():
    testpath = sunpy.data.test.rootdir
    return sunpy.map.Map(Path(testpath) / "aia_171_level1.fits")


@pytest.fixture()
def aia171_test_shift():
    return np.array([3, 5])


@pytest.fixture()
def aia171_test_map_layer(aia171_test_map):
    return aia171_test_map.data.astype("float32")  # SciPy 1.4 requires at least 16-bit floats


@pytest.fixture()
def aia171_test_map_layer_shape(aia171_test_map_layer):
    return aia171_test_map_layer.shape


@pytest.fixture()
def aia171_test_template(aia171_test_shift, aia171_test_map_layer, aia171_test_map_layer_shape):
    # Test template
    a1 = aia171_test_shift[0] + aia171_test_map_layer_shape[0] // 4
    a2 = aia171_test_shift[0] + 3 * aia171_test_map_layer_shape[0] // 4
    b1 = aia171_test_shift[1] + aia171_test_map_layer_shape[1] // 4
    b2 = aia171_test_shift[1] + 3 * aia171_test_map_layer_shape[1] // 4
    return aia171_test_map_layer[a1:a2, b1:b2]


@pytest.fixture()
def aia171_test_template_shape(aia171_test_template):
    return aia171_test_template.shape
