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

    fig = plt.figure()
    out.plot()
