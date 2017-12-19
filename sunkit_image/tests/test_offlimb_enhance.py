#
# Test the off limb enhancement code
#
from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_allclose
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
    return np.arange(1, 1.5, 0.1) * u.R_sun


@pytest.fixture
def map_r(aia_map):
    """
    Load SunPy's test AIA image.
    """
    return find_pixel_radii(aia_map)


def test_radial_bins(radial_bins, map_r):

    # Test the default radial bins
    rb = _radial_bins(None, map_r)
    assert len(rb) == 50
    assert rb.unit is u.R_sun
    assert_allclose((rb[1] - rb[0]).value, 0.01)
    assert_allclose(rb[0].value, 1.0)
    assert_allclose(rb[-1].value, np.max(map_r).value)

    # Test another set of
    rb2 = _radial_bins(radial_bins, map_r)
    assert len(rb2) == 5
    assert rb2.unit is u.R_sun
    assert_allclose((rb2[1] - rb2[0]).value, 0.1)
    assert_allclose(rb2[0].value, 1.0)
    assert_allclose(rb2[-1].value, 1.5)


def test_bin_edge_summary(radial_bins, binfit='left'):

    # Test
    bes_left = bin_edge_summary(radial_bins, binfit='left')
    assert len(bes_left)
    assert bes_left[0] == 1 * u.

    # Test
    bes_right = bin_edge_summary(radial_bins, binfit='right')
    assert len(bes_right)

    # Test
    bes_center = bin_edge_summary(radial_bins, binfit='center')
    assert len(bes_center)

    if binfit == 'center':
        rfit = 0.5 * (r[0:-1] + r[1:])
    elif binfit == 'left':
        rfit = r[0:-1]
    elif binfit == 'right':
        rfit = r[1:]
    else:
        raise ValueError(
            'Keyword "binfit" must have value "center", "left" or "right"')
    return rfit

