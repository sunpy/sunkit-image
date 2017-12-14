#
# Test the off limb enhancement code
#
from __future__ import absolute_import

import numpy as np
import astropy.units as u
import sunpy
import sunpy.map
from sunpy.util.metadata import MetaDict
import pytest
import os
import sunpy.data.test


@pytest.fixture
def aia_map():
    """
    Load SunPy's test AIA image.
    """
    testpath = sunpy.data.test.rootdir
    aia_file = os.path.join(testpath, "aia_171_level1.fits")
    return sunpy.map.Map(aia_file)


@pytest.fixture
def radial_bins():
    """
    Load SunPy's test AIA image.
    """
    return np.arange(1, 1.5, 0.01) * u.R_sun


def test_radial_bins(r, map_r):
    """

    Parameters
    ----------
    r
    map_r

    Returns
    -------

    """
    if r is None:
        return np.arange(1, np.max(map_r).to(u.R_sun).value, 0.01) * u.R_sun
    else:
        return r.to(u.R_sun)
