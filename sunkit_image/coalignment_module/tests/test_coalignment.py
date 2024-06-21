import astropy.units as u
import numpy as np
import pytest
from scipy.ndimage import shift as sp_shift
from sunpy.map import Map


@pytest.fixture()
def aia171_test_mc_pixel_displacements():
    return np.asarray([1.6, 10.1])


@pytest.fixture()
def aia171_mc_arcsec_displacements(aia171_test_mc_pixel_displacements, aia171_test_map):
    return {
        "x": np.asarray([0.0, aia171_test_mc_pixel_displacements[1] * aia171_test_map.scale[0].value]) * u.arcsec,
        "y": np.asarray([0.0, aia171_test_mc_pixel_displacements[0] * aia171_test_map.scale[1].value]) * u.arcsec,
    }


@pytest.fixture()
def aia171_test_shifted_map(aia171_test_map, aia171_test_map_layer, aia171_test_mc_pixel_displacements):
    # Create a map that has been shifted a known amount.
    d1 = sp_shift(aia171_test_map_layer, aia171_test_mc_pixel_displacements)
    # return the map
    return Map(d1, aia171_test_map.meta)
