import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import sunpy.map

from sunkit_image.coalignment import coalign
from sunkit_image.coalignment.interface import AffineParams, _update_fits_wcs_metadata
from sunkit_image.tests.helpers import figure_test

POINTING_SHIFT = [25, 50] * u.arcsec

@pytest.fixture()
def eis_test_map():
    url = "https://github.com/sunpy/data/raw/main/sunkit-image/eis_20140108_095727.fe_12_195_119.2c-0.int.fits"
    with fits.open(url) as hdul:
        return sunpy.map.Map(hdul[0].data, hdul[0].header)


@pytest.fixture()
def aia193_test_map():
    # This is matched to the EIS observation time
    url = "https://github.com/sunpy/data/raw/refs/heads/main/sunkit-image/aia.lev1.193A_2014_01_08T09_57_30.84Z.image_lev1.fits"
    with fits.open(url) as hdul:
        hdul.verify('silentfix')
        return sunpy.map.Map(hdul[1].data, hdul[1].header)


@pytest.mark.remote_data()
def test_coalignment_eis_aia(eis_test_map, aia193_test_map):
    nx = (aia193_test_map.scale.axis1 * aia193_test_map.dimensions.x) / eis_test_map.scale[0]
    ny = (aia193_test_map.scale.axis2 * aia193_test_map.dimensions.y) / eis_test_map.scale[1]
    aia193_test_downsampled_map = aia193_test_map.resample(u.Quantity([nx, ny]))
    coaligned_eis_map = coalign(eis_test_map, aia193_test_downsampled_map, method='match_template')
    # Check that correction is as expected based on known pointing offset
    assert u.allclose(
        eis_test_map.reference_coordinate.separation(coaligned_eis_map.reference_coordinate),
        5.95935177*u.arcsec,
    )


@pytest.fixture()
def cutout_map(aia171_test_map):
    bottom_left = SkyCoord(200*u.arcsec, 100*u.arcsec, frame=aia171_test_map.coordinate_frame)
    top_right = SkyCoord(600*u.arcsec, 900*u.arcsec, frame=aia171_test_map.coordinate_frame)
    return aia171_test_map.submap(bottom_left, top_right=top_right)


@pytest.fixture
def incorrect_pointing_map(aia171_test_map):
    return aia171_test_map.shift_reference_coord(*POINTING_SHIFT)


@pytest.fixture()
def incorrect_pointing_cutout_map(cutout_map):
    return cutout_map.shift_reference_coord(*POINTING_SHIFT)


def test_coalignment_match_template(incorrect_pointing_cutout_map, aia171_test_map):
    fixed_cutout_map = coalign(incorrect_pointing_cutout_map, aia171_test_map)
    # The actual shifts applied by coalignment should be equal to the expected shifts
    u.allclose(
        np.fabs(incorrect_pointing_cutout_map.reference_coordinate.Tx-fixed_cutout_map.reference_coordinate.Tx),
        POINTING_SHIFT[0],
    )
    u.allclose(
        np.fabs(incorrect_pointing_cutout_map.reference_coordinate.Ty-fixed_cutout_map.reference_coordinate.Ty),
        POINTING_SHIFT[1],
    )


def test_coalign_phase_cross_correlation(incorrect_pointing_map, aia171_test_map):
    fixed_map = coalign(incorrect_pointing_map, aia171_test_map, method='phase_cross_correlation')
    # The actual shifts applied by coalignment should be equal to the expected shifts
    u.allclose(
        np.fabs(incorrect_pointing_map.reference_coordinate.Tx-fixed_map.reference_coordinate.Tx),
        POINTING_SHIFT[0],
    )
    u.allclose(
        np.fabs(incorrect_pointing_map.reference_coordinate.Ty-fixed_map.reference_coordinate.Ty),
        POINTING_SHIFT[1],
    )


@figure_test
def test_coalignment_figure(incorrect_pointing_cutout_map, cutout_map, aia171_test_map):
    levels = [200, 400, 500, 700, 800]*cutout_map.unit
    fixed_cutout_map = coalign(incorrect_pointing_cutout_map, aia171_test_map)
    fig = plt.figure(figsize=(10, 7.5))
    # Messed up map
    ax = fig.add_subplot(121, projection=incorrect_pointing_cutout_map)
    incorrect_pointing_cutout_map.plot(axes=ax, title='Incorrect Pointing')
    cutout_map.draw_contours(levels, axes=ax, alpha=0.3)
    # After coalignment
    ax = fig.add_subplot(122, projection=fixed_cutout_map)
    fixed_cutout_map.plot(axes=ax, title='Fixed Pointing')
    cutout_map.draw_contours(levels, axes=ax, alpha=0.3)
    fig.tight_layout()
    return fig


def test_unsupported_affine_parameters(incorrect_pointing_cutout_map, aia171_test_map):
    affine_rot = AffineParams(
        scale=[1,1],
        rotation_matrix=2*np.eye(2),
        translation=[0,0],
    )
    with pytest.raises(NotImplementedError, match=r"Changes to the rotation metadata are currently not supported."):
        _update_fits_wcs_metadata(incorrect_pointing_cutout_map, aia171_test_map, affine_rot)
    affine_scale = AffineParams(
        scale=[2,3],
        rotation_matrix=np.eye(2),
        translation=[0,0],
    )
    with pytest.raises(NotImplementedError, match=r"Changes to the pixel scale metadata are currently not supported."):
        _update_fits_wcs_metadata(incorrect_pointing_cutout_map, aia171_test_map, affine_scale)
