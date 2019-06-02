import numpy as np
import matplotlib.pyplot as plt
import pytest

import sunpy.map
import sunpy.data.sample
from sunpy.tests.helpers import figure_test

import sunkit_image.enhance as enhance


@pytest.fixture
@pytest.mark.remotedata
def smap():
    return sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

@pytest.fixture
def test_map():
    map_test = [[1.,1.,1.,1.],
                [1.,5.,5.,1.],
                [1.,5.,5.,1.],
                [1.,1.,1.,1.]]
    
    return np.array(map_test)


def test_background_supression(test_map):
    
    expect = [[0.,0.,0.,0.],
              [0.,5.,5.,0.],
              [0.,5.,5.,0.],
              [0.,0.,0.,0.]]
    
    result = enhance.background_supression(test_map, 2, 0)
    assert np.allclose(expect, result)

    expect = [[1.,1.,1.,1.],
              [1.,5.,5.,1.],
              [1.,5.,5.,1.],
              [1.,1.,1.,1.]]
    
    result = enhance.background_supression(test_map, 2)
    assert np.allclose(expect, result)


@pytest.fixture
def image():
    return np.ones((4,4),dtype=float)

def test_bandpass_filter(image):

    expect = np.zeros((4,4))

    result = enhance.bandpass_filter(test_map)

    assert np.allclose(expect, result)
    