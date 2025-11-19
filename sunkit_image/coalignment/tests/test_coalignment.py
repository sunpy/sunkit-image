import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose

import sunpy.map
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from sunpy.util.exceptions import SunpyUserWarning

from sunkit_image.coalignment import coalign
from sunkit_image.coalignment.interface import AffineParams, _update_fits_wcs_metadata
from sunkit_image.tests.helpers import figure_test

POSITIVE_POINTING_SHIFT = [25, 50] * u.arcsec
NEGATIVE_POINTING_SHIFT = [-25, -50] * u.arcsec
MIXED_POINTING_SHIFT = [25, -50] * u.arcsec

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


@pytest.fixture()
def incorrect_shifted_once_map(aia171_test_map):
    return aia171_test_map.shift_reference_coord(30*u.arcsec, -40*u.arcsec)


@pytest.fixture()
def cutout_map(aia171_test_map):
    bottom_left = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=aia171_test_map.coordinate_frame)
    top_right = SkyCoord(900*u.arcsec, 900*u.arcsec, frame=aia171_test_map.coordinate_frame)
    return aia171_test_map.submap(bottom_left, top_right=top_right)


@pytest.fixture(params=[
    POSITIVE_POINTING_SHIFT,
    NEGATIVE_POINTING_SHIFT,
    MIXED_POINTING_SHIFT,
])
def incorrect_pointing_map_and_shift(request, aia171_test_map):
    return aia171_test_map.shift_reference_coord(*request.param), request.param


@pytest.fixture(params=[
    POSITIVE_POINTING_SHIFT,
    NEGATIVE_POINTING_SHIFT,
    MIXED_POINTING_SHIFT,
])
def incorrect_pointing_cutout_map_and_shift(request, cutout_map):
    return cutout_map.shift_reference_coord(*request.param), request.param


@pytest.mark.remote_data()
def test_coalignment_eis_aia(eis_test_map, aia193_test_map):
    nx = (aia193_test_map.scale.axis1 * aia193_test_map.dimensions.x) / eis_test_map.scale[0]
    ny = (aia193_test_map.scale.axis2 * aia193_test_map.dimensions.y) / eis_test_map.scale[1]
    aia193_test_downsampled_map = aia193_test_map.resample(u.Quantity([nx, ny]))
    coaligned_eis_map = coalign(eis_test_map, aia193_test_downsampled_map, method='match_template')
    # Check that correction is as expected based on known pointing offset
    assert_quantity_allclose(
        eis_test_map.reference_coordinate.separation(coaligned_eis_map.reference_coordinate),
        5.95935177*u.arcsec,
    )


def test_coalignment_match_template_full_map(incorrect_pointing_map_and_shift, aia171_test_map):
    incorrect_pointing_map, pointing_shift = incorrect_pointing_map_and_shift

    # Crop out the array that is outside the solar disk to have something to align to
    hpc_coords = all_coordinates_from_map(incorrect_pointing_map)
    mask = coordinate_is_on_solar_disk(hpc_coords)
    masked_map = sunpy.map.Map(np.abs(incorrect_pointing_map.data*mask), incorrect_pointing_map.meta)

    # We have to pad the input map because otherwise match_template cannot find a good match
    fixed_map = coalign(masked_map, aia171_test_map, pad_input=True)

    assert_quantity_allclose(
        (incorrect_pointing_map.reference_coordinate.Tx-fixed_map.reference_coordinate.Tx),
        pointing_shift[0],
        atol=1*u.arcsec,
    )
    assert_quantity_allclose(
        (incorrect_pointing_map.reference_coordinate.Ty-fixed_map.reference_coordinate.Ty),
        pointing_shift[1],
        atol=1*u.arcsec,
    )


def test_coalignment_match_template_cutout(incorrect_pointing_cutout_map_and_shift, aia171_test_map):
    incorrect_pointing_cutout_map, pointing_shift = incorrect_pointing_cutout_map_and_shift
    fixed_cutout_map = coalign(incorrect_pointing_cutout_map, aia171_test_map)
    assert_quantity_allclose(
        (incorrect_pointing_cutout_map.reference_coordinate.Tx-fixed_cutout_map.reference_coordinate.Tx),
        pointing_shift[0],
        atol=1*u.arcsec,
    )
    assert_quantity_allclose(
        (incorrect_pointing_cutout_map.reference_coordinate.Ty-fixed_cutout_map.reference_coordinate.Ty),
        pointing_shift[1],
        atol=1*u.arcsec,
    )


def test_coalign_phase_cross_correlation(incorrect_pointing_map_and_shift, aia171_test_map):
    incorrect_pointing_map, pointing_shift = incorrect_pointing_map_and_shift
    fixed_map = coalign(incorrect_pointing_map, aia171_test_map, method='phase_cross_correlation')
    assert_quantity_allclose(
        (incorrect_pointing_map.reference_coordinate.Tx-fixed_map.reference_coordinate.Tx),
        pointing_shift[0],
        atol=1*u.arcsec,
    )
    assert_quantity_allclose(
        (incorrect_pointing_map.reference_coordinate.Ty-fixed_map.reference_coordinate.Ty),
        pointing_shift[1],
        atol=1*u.arcsec,
    )


@figure_test
def test_coalignment_figure(incorrect_pointing_cutout_map_and_shift, cutout_map, aia171_test_map):
    # This is three separate figure tests
    levels = [200, 800]*cutout_map.unit
    incorrect_pointing_cutout_map, _ = incorrect_pointing_cutout_map_and_shift
    fixed_cutout_map = coalign(incorrect_pointing_cutout_map, aia171_test_map)
    fig = plt.figure(figsize=(10, 7.5))
    # Messed up map
    ax = fig.add_subplot(121, projection=incorrect_pointing_cutout_map)
    incorrect_pointing_cutout_map.plot(axes=ax, title='Incorrect Pointing')
    cutout_map.draw_contours(levels, axes=ax)
    # After coalignment
    ax = fig.add_subplot(122, projection=fixed_cutout_map)
    fixed_cutout_map.plot(axes=ax, title='Fixed Pointing')
    cutout_map.draw_contours(levels, axes=ax)
    fig.tight_layout()
    return fig


def test_unsupported_affine_parameters(incorrect_shifted_once_map, aia171_test_map):
    affine_rot = AffineParams(
        scale=[1,1],
        rotation_matrix=2*np.eye(2),
        translation=[0,0],
    )
    with pytest.raises(NotImplementedError, match=r"Changes to the rotation metadata are currently not supported."):
        _update_fits_wcs_metadata(incorrect_shifted_once_map, aia171_test_map, affine_rot)
    affine_scale = AffineParams(
        scale=[2,3],
        rotation_matrix=np.eye(2),
        translation=[0,0],
    )
    with pytest.raises(NotImplementedError, match=r"Changes to the pixel scale metadata are currently not supported."):
        _update_fits_wcs_metadata(incorrect_shifted_once_map, aia171_test_map, affine_scale)


def test_warnings_coalign(incorrect_shifted_once_map, aia171_test_map):
    time_shift_meta = incorrect_shifted_once_map.meta.copy()
    time_shift_meta['DATE-OBS'] = '2014-01-08T09:57:30.84'
    time_shift = sunpy.map.Map(incorrect_shifted_once_map.data, time_shift_meta)
    with pytest.warns(SunpyUserWarning, match=r"The difference in observation times of the reference and target maps is large."):
        with pytest.raises(ValueError, match=r"match_template returned an array with fewer than two values,"):
            coalign(time_shift, aia171_test_map)

    resampled_map = incorrect_shifted_once_map.resample([100,100]*u.pix)
    with pytest.warns(SunpyUserWarning, match=r"The reference and target maps have different plate scales."):
        coalign(resampled_map, aia171_test_map)
