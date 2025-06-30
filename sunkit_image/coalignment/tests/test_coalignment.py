import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

from sunkit_image.coalignment import coalign
from sunkit_image.coalignment.interface import REGISTERED_METHODS, AffineParams, register_coalignment_method
from sunkit_image.tests.helpers import figure_test


@pytest.fixture()
def eis_test_map():
    url = "https://github.com/sunpy/data/raw/main/sunkit-image/eis_20140108_095727.fe_12_195_119.2c-0.int.fits"
    with fits.open(url) as hdul:
        return sunpy.map.Map(hdul[0].data, hdul[0].header)


@pytest.fixture()
def aia193_test_map(eis_test_map):
    query = Fido.search(
        a.Time(start=eis_test_map.date-1*u.minute,
               end=eis_test_map.date+1*u.minute,
               near=eis_test_map.date),
        a.Instrument.aia,
        a.Wavelength(193*u.angstrom),
    )
    file = Fido.fetch(query)
    return sunpy.map.Map(file)


@pytest.mark.remote_data()
def test_coalignment(eis_test_map, aia193_test_map):
    nx = (aia193_test_map.scale.axis1 * aia193_test_map.dimensions.x) / eis_test_map.scale[0]
    ny = (aia193_test_map.scale.axis2 * aia193_test_map.dimensions.y) / eis_test_map.scale[1]
    aia193_test_downsampled_map = aia193_test_map.resample(u.Quantity([nx, ny]))
    coaligned_eis_map = coalign(eis_test_map, aia193_test_downsampled_map, "match_template")
    assert_allclose(coaligned_eis_map.wcs.wcs.crval[0],
                    aia193_test_downsampled_map.wcs.wcs.crval[0],
                    rtol = 1e-2,
                    atol = 0.13)
    assert_allclose(coaligned_eis_map.wcs.wcs.crval[1],
                    aia193_test_downsampled_map.wcs.wcs.crval[1],
                    rtol = 1e-2,
                    atol = 0.13)


@pytest.fixture()
def cutout_map(aia171_test_map):
    aia_map = sunpy.map.Map(aia171_test_map)
    bottom_left = SkyCoord(-300 * u.arcsec, -300 * u.arcsec, frame = aia_map.coordinate_frame)
    top_right = SkyCoord(800 * u.arcsec, 600 * u.arcsec, frame = aia_map.coordinate_frame)
    return aia_map.submap(bottom_left, top_right=top_right)


@pytest.fixture()
def incorrect_pointing_map(cutout_map):
    return cutout_map.shift_reference_coord(25 * u.arcsec, 50 * u.arcsec)


def test_coalignment_reflects_pixel_shifts(incorrect_pointing_map, aia171_test_map):
    """
    Check if coalignment adjusts world coordinates as expected based on
    reference coordinate shifts.
    """
    original_world_coords = cutout_map.reference_coordinate
    fixed_cutout_map = coalign(incorrect_pointing_map, aia171_test_map)
    fixed_world_coords = fixed_cutout_map.reference_coordinate
    # The actual shifts applied by coalignment should be equal to the expected shifts
    assert_allclose(original_world_coords.Tx, fixed_world_coords.Tx, rtol=1e-2, atol=0.4)
    assert_allclose(original_world_coords.Ty, fixed_world_coords.Ty, rtol=1e-2, atol=0.4)


@figure_test
def test_coalignment_figure(incorrect_pointing_map, aia171_test_map):
    levels = [200, 400, 500, 700, 800] * cutout_map.unit
    fixed_cutout_map = coalign(aia171_test_map, incorrect_pointing_map)
    fig = plt.figure(figsize=(15, 7.5))
    # Before coalignment
    ax1 = fig.add_subplot(131, projection=cutout_map)
    cutout_map.plot(axes=ax1, title='Original Cutout')
    cutout_map.draw_contours(levels, axes=ax1, alpha=0.3)
    # Messed up map
    ax2 = fig.add_subplot(132, projection=incorrect_pointing_map)
    incorrect_pointing_map.plot(axes=ax2, title='Messed Cutout')
    cutout_map.draw_contours(levels, axes=ax2, alpha=0.3)
    # After coalignment
    ax3 = fig.add_subplot(133, projection=fixed_cutout_map)
    fixed_cutout_map.plot(axes=ax3, title='Fixed Cutout')
    cutout_map.draw_contours(levels, axes=ax3, alpha=0.3)
    fig.tight_layout()
    return fig


@pytest.fixture()
@register_coalignment_method("scaling")
def coalign_scaling(reference_map, target_map):
    return AffineParams(scale=np.array([0.25, 0.25]), rotation_matrix=np.eye(2), translation=(0 , 0))


def test_coalignment_reflects_scaling(cutout_map, aia171_test_map):
    scale_factor = 4
    rescaled_map = cutout_map.resample(u.Quantity([cutout_map.dimensions.x * scale_factor, cutout_map.dimensions.y * scale_factor]))
    fixed_cutout_map = coalign(aia171_test_map, rescaled_map, method="scaling")
    assert_allclose(fixed_cutout_map.scale[0].value, cutout_map.scale[0].value, rtol=1e-5, atol=0)
    assert_allclose(fixed_cutout_map.scale[1].value, cutout_map.scale[1].value, rtol=1e-5, atol=0)


@pytest.fixture()
@register_coalignment_method("rotation")
def coalign_rotation(reference_map, target_map):
    rotation_matrix = np.array([[0.866, -0.5], [0.5, 0.866]])
    return AffineParams(scale=np.array([1.0, 1.0]), rotation_matrix= rotation_matrix, translation=(0 , 0))


def test_coalignment_reflects_rotation(cutout_map, aia171_test_map):
    rotation_angle = 30 * u.deg
    rotated_map = cutout_map.rotate(angle=rotation_angle)
    fixed_cutout_map = coalign(aia171_test_map, rotated_map, method="rotation")
    assert_allclose(fixed_cutout_map.rotation_matrix[0, 0], cutout_map.rotation_matrix[0, 0], rtol=1e-4, atol=0)
    assert_allclose(fixed_cutout_map.rotation_matrix[1, 1], cutout_map.rotation_matrix[1, 1], rtol=1e-4, atol=0)


def test_register_coalignment_method():
    @register_coalignment_method("test_method")
    def test_func():
        return "Test function"

    assert "test_method" in REGISTERED_METHODS
    assert REGISTERED_METHODS["test_method"] == test_func
    assert test_func() == "Test function"
