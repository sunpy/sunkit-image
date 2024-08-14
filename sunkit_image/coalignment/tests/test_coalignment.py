import copy
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt
import pytest
from scipy.ndimage import shift as sp_shift

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch

import sunpy.map
from sunpy.net import Fido, attrs as a

from sunkit_image.coalignment import coalign
from sunkit_image.data.test import get_test_filepath
from sunkit_image.tests.helpers import figure_test


@pytest.fixture()
def is_test_map():
    return sunpy.map.Map(get_test_filepath("eis_20140108_095727.fe_12_195_119.2c-0.int.fits"))


@pytest.mark.remote_data()
def aia193_test_map():
    date_start = "2014-01-08T09:57:27"
    date_end = "2014-01-08T10:00:00"
    aia_map = Fido.fetch(Fido.search(a.Time(start=date_start, end=date_end), a.Instrument('aia'), a.Wavelength(193 * u.angstrom)))
    return sunpy.map.Map(aia_map)


@pytest.mark.remote_data()
def aia193_test_downsampled_map(is_test_map, aia193_test_map):
    nx = (aia193_test_map.scale.axis1 * aia193_test_map.dimensions.x) / is_test_map.scale.axis1
    ny = (aia193_test_map.scale.axis2 * aia193_test_map.dimensions.y) / is_test_map.scale.axis2
    return aia193_test_map.resample(u.Quantity([nx, ny]))


@pytest.mark.remote_data()
def aia193_test_shifted_map(aia193_test_map):
    pixel_displacements = np.array([5.0, 5.0])
    shifted_data = sp_shift(aia193_test_map.data, pixel_displacements)
    return sunpy.map.Map(shifted_data, aia193_test_map.meta)


@pytest.mark.remote_data()
def test_coalignment(is_test_map, aia193_test_downsampled_map):
    coaligned_is_map = coalign(aia193_test_downsampled_map, is_test_map, "match_template")
    assert coaligned_is_map.data.shape == is_test_map.data.shape
    assert coaligned_is_map.wcs.wcs.crval[0] == aia193_test_downsampled_map.wcs.wcs.crval[0]
    assert coaligned_is_map.wcs.wcs.crval[1] == aia193_test_downsampled_map.wcs.wcs.crval[1]


@pytest.fixture()
def cutout_map(aia171_test_map):
    aia_map = sunpy.map.Map(aia171_test_map)
    bottom_left = SkyCoord(300 * u.arcsec, -300 * u.arcsec, frame = aia_map.coordinate_frame)
    top_right = SkyCoord(800 * u.arcsec, 200 * u.arcsec, frame = aia_map.coordinate_frame)
    cutout_map = aia_map.submap(bottom_left, top_right=top_right)
    return cutout_map


def test_coalignment_reflects_pixel_shifts(cutout_map, aia171_test_map):
    """Check if coalignment adjusts world coordinates as expected based on pixel shifts."""
    dx_pix, dy_pix = 10, 10  # Considering the pixel shifts are small enough to not cause any distortion
    messed_map = copy.deepcopy(cutout_map)
    messed_map.meta['crpix1'] += dx_pix
    messed_map.meta['crpix2'] += dy_pix

    original_world_coords = cutout_map.reference_coordinate
    shifted_world_coords = cutout_map.wcs.pixel_to_world(
        cutout_map.reference_pixel.x.value + dx_pix,
        cutout_map.reference_pixel.y.value + dy_pix
    )
    expected_shift_Tx = shifted_world_coords.Tx - original_world_coords.Tx
    expected_shift_Ty = shifted_world_coords.Ty - original_world_coords.Ty
    # Fix the messed map
    fixed_cutout_map = coalign(aia171_test_map, messed_map)

    fixed_world_coords = fixed_cutout_map.reference_coordinate
    actual_shift_Tx = fixed_world_coords.Tx - original_world_coords.Tx
    actual_shift_Ty = fixed_world_coords.Ty - original_world_coords.Ty
    # The shifts should be equal to the expected shifts
    assert_allclose(actual_shift_Tx, expected_shift_Tx, rtol=1e-2, atol=0)
    assert_allclose(actual_shift_Ty, expected_shift_Ty, rtol=1e-2, atol=0)

    assert not np.allclose(original_world_coords.Tx, fixed_world_coords.Tx)
    assert not np.allclose(original_world_coords.Ty, fixed_world_coords.Ty)


@figure_test
def test_coalignment_figure(cutout_map, aia171_test_map):
    levels = [200, 400, 500, 700, 800] * cutout_map.unit
    messed_map = copy.deepcopy(cutout_map)
    messed_map.meta['crpix1'] += 10
    messed_map.meta['crpix2'] += 10
    fixed_cutout_map = coalign(aia171_test_map, messed_map)
    fig = plt.figure(figsize=(15, 7.5))
    # Before coalignment
    ax1 = fig.add_subplot(121, projection=cutout_map)
    cutout_map.plot(axes=ax1, title='Original Cutout', cmap='Blues_r', norm=ImageNormalize(stretch=AsinhStretch()))
    cutout_map.draw_contours(levels, axes=ax1, alpha=0.3)
    # After coalignment
    ax2 = fig.add_subplot(122, projection=fixed_cutout_map)
    fixed_cutout_map.plot(axes=ax2, title='Fixed Cutout', cmap='Blues_r', norm=ImageNormalize(stretch=AsinhStretch()))
    fixed_cutout_map.draw_contours(levels, axes=ax2, alpha=0.3)
    fig.tight_layout()
    return fig