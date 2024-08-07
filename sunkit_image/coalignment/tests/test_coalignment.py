import numpy as np
import pytest
from scipy.ndimage import shift as sp_shift

import astropy.units as u

from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a

from sunkit_image.coalignment import coalign
from sunkit_image.data.test import get_test_filepath


@pytest.fixture()
def is_test_map():
    return sunpy.map.Map(get_test_filepath("is_20140108_095727.fe_12_195_119.2c-0.int.fits"))

@pytest.fixture()
def aia193_test_map():
    date_start = "2014-01-08T09:57:27"
    date_end = "2014-01-08T10:00:00"
    aia_map = Fido.fetch(Fido.search(a.Time(start=date_start, end=date_end), a.Instrument('aia'), a.Wavelength(193 * u.angstrom)))
    return sunpy.map.Map(aia_map)

@pytest.fixture()
def aia193_test_downsampled_map(is_test_map, aia193_test_map):
    nx = (aia193_test_map.scale.axis1 * aia193_test_map.dimensions.x) / is_test_map.scale.axis1
    ny = (aia193_test_map.scale.axis2 * aia193_test_map.dimensions.y) / is_test_map.scale.axis2
    return aia193_test_map.resample(u.Quantity([nx, ny]))

@pytest.fixture()
def aia193_test_shifted_map(aia193_test_map):
    pixel_displacements = np.array([5.0, 5.0])
    shifted_data = sp_shift(aia193_test_map.data, pixel_displacements)
    return Map(shifted_data, aia193_test_map.meta)

def test_coalignment(is_test_map, aia193_test_downsampled_map):
    coaligned_is_map = coalign(aia193_test_downsampled_map, is_test_map, "match_template")

    # Assertions to ensure the maps are aligned
    assert coaligned_is_map.data.shape == is_test_map.data.shape
    assert coaligned_is_map.wcs.wcs.crval[0] == aia193_test_downsampled_map.wcs.wcs.crval[0]
    assert coaligned_is_map.wcs.wcs.crval[1] == aia193_test_downsampled_map.wcs.wcs.crval[1]

def test_shifted_map_alignment(is_test_map, aia193_test_shifted_map):
    coaligned_is_map = coalign(aia193_test_shifted_map, is_test_map, "match_template")

    # Ensure the alignment is corrected by comparing with the unshifted map
    assert coaligned_is_map.data.shape == is_test_map.data.shape
    assert coaligned_is_map.wcs.wcs.crval[0] == aia193_test_shifted_map.wcs.wcs.crval[0]
    assert coaligned_is_map.wcs.wcs.crval[1] == aia193_test_shifted_map.wcs.wcs.crval[1]
