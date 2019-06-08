import numpy as np
import matplotlib.pyplot as plt
import pytest

import sunpy.map
import sunpy.data.sample
from sunpy.tests.helpers import figure_test

import sunkit_image.enhance as enhance


@pytest.fixture
@pytest.mark.remote_data
def smap():
    return sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)


@figure_test
@pytest.mark.remote_data
def test_mgn(smap):

    out = enhance.mgn(smap.data)
    out = sunpy.map.Map(out, smap.meta)

    out.plot()


@pytest.fixture
def map_test():

    A = np.array([[1., 1., 1., 1.],
                 [1., 5., 5., 1.],
                 [1., 5., 5., 1.],
                 [1., 1., 1., 1.]])
    return A


def test_multiscale_gaussian(map_test):

    # Assuming the algorithm works fine then the below two should be equal.
    expect1 = enhance.mgn(map_test, [1])
    expect2 = enhance.mgn(map_test, [1, 1])

    assert np.allclose(expect1, expect2)
