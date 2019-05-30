import numpy as np
import matplotlib.pyplot as plt
import pytest

import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE
from sunpy.tests.helpers import figure_test

import sunkit_image.enhance as enhance


@pytest.fixture
def smap():
    return sunpy.map.Map(AIA_171_IMAGE)


@figure_test
def test_mgn(smap):
    
    out = enhance.mgn(smap.data)
    out = sunpy.map.Map(out, smap.meta)

    fig = plt.figure()
    out.plot()