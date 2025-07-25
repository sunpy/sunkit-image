import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.ndimage import shift as sp_shift

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.data.test
from sunpy.map import Map, MapSequence
from sunpy.util import SunpyUserWarning

from sunkit_image.coalignment import (
    _calculate_clipping,
    _check_for_nonfinite_entries,
    _clip_edges,
    _default_fmap_function,
    _find_best_match_location,
    _get_correlation_shifts,
    _lower_clip,
    _parabolic_turning_point,
    _upper_clip,
    apply_shifts,
    calculate_match_template_shift,
    mapsequence_coalign_by_match_template,
    match_template_to_layer,
)


@pytest.fixture()
def aia171_test_clipping():
    return np.asarray([0.2, -0.3, -1.0001])


@pytest.fixture()
def aia171_test_map():
    testpath = sunpy.data.test.rootdir
    return sunpy.map.Map(Path(testpath) / "aia_171_level1.fits")


@pytest.fixture()
def aia171_test_shift():
    return np.array([3, 5])


@pytest.fixture()
def aia171_test_map_layer(aia171_test_map):
    return aia171_test_map.data.astype("float32")  # SciPy 1.4 requires at least 16-bit floats


@pytest.fixture()
def aia171_test_map_layer_shape(aia171_test_map_layer):
    return aia171_test_map_layer.shape


@pytest.fixture()
def aia171_test_template(aia171_test_shift, aia171_test_map_layer, aia171_test_map_layer_shape):
    # Test template
    a1 = aia171_test_shift[0] + aia171_test_map_layer_shape[0] // 4
    a2 = aia171_test_shift[0] + 3 * aia171_test_map_layer_shape[0] // 4
    b1 = aia171_test_shift[1] + aia171_test_map_layer_shape[1] // 4
    b2 = aia171_test_shift[1] + 3 * aia171_test_map_layer_shape[1] // 4
    return aia171_test_map_layer[a1:a2, b1:b2]


@pytest.fixture()
def aia171_test_template_shape(aia171_test_template):
    return aia171_test_template.shape


def test_parabolic_turning_point():
    assert _parabolic_turning_point(np.asarray([6.0, 2.0, 0.0])) == 1.5


def test_check_for_nonfinite_entries():
    with warnings.catch_warnings(record=True) as warning_list:
        a = np.zeros((3, 3))
        b = np.ones((3, 3))
        _check_for_nonfinite_entries(a, b)

    assert len(warning_list) == 0

    for i in range(9):
        for non_number in [np.nan, np.inf]:
            a = np.ones(9)
            a[i] = non_number
            b = a.reshape(3, 3)

            with pytest.warns(SunpyUserWarning, match="The layer image has nonfinite entries."):
                _check_for_nonfinite_entries(b, np.ones((3, 3)))

            with pytest.warns(SunpyUserWarning, match="The template image has nonfinite entries."):
                _check_for_nonfinite_entries(np.ones((3, 3)), b)

            with pytest.warns(SunpyUserWarning, match="The layer image has nonfinite entries."):
                with pytest.warns(SunpyUserWarning, match="The template image has nonfinite entries."):
                    _check_for_nonfinite_entries(b, b)


def test_match_template_to_layer(
    aia171_test_map_layer,
    aia171_test_template,
    aia171_test_map_layer_shape,
    aia171_test_template_shape,
):
    result = match_template_to_layer(aia171_test_map_layer, aia171_test_template)
    assert_allclose(
        result.shape[0],
        aia171_test_map_layer_shape[0] - aia171_test_template_shape[0] + 1,
    )
    assert_allclose(
        result.shape[1],
        aia171_test_map_layer_shape[1] - aia171_test_template_shape[1] + 1,
    )
    assert_allclose(np.max(result), 1.00, rtol=1e-2, atol=0)


def test_get_correlation_shifts():
    # Input array is 3 by 3, the most common case
    test_array = np.zeros((3, 3))
    test_array[1, 1] = 1
    test_array[2, 1] = 0.6
    test_array[1, 2] = 0.2
    y_test, x_test = _get_correlation_shifts(test_array)
    assert_allclose(y_test.value, 0.214285714286, rtol=1e-2, atol=0)
    assert_allclose(x_test.value, 0.0555555555556, rtol=1e-2, atol=0)

    # Input array is smaller in one direction than the other.
    test_array = np.zeros((2, 2))
    test_array[0, 0] = 0.1
    test_array[0, 1] = 0.2
    test_array[1, 0] = 0.4
    test_array[1, 1] = 0.3
    y_test, x_test = _get_correlation_shifts(test_array)
    assert_allclose(y_test.value, 1.0, rtol=1e-2, atol=0)
    assert_allclose(x_test.value, 0.0, rtol=1e-2, atol=0)

    # Input array is too big in either direction
    test_array = np.zeros((4, 3))
    with pytest.raises(ValueError, match="Input array dimension should not be greater than 3 in any dimension."):
        _get_correlation_shifts(test_array)

    test_array = np.zeros((3, 4))
    with pytest.raises(ValueError, match="Input array dimension should not be greater than 3 in any dimension."):
        _get_correlation_shifts(test_array)


def test_find_best_match_location(aia171_test_map_layer, aia171_test_template, aia171_test_shift):
    result = match_template_to_layer(aia171_test_map_layer, aia171_test_template)
    match_location = u.Quantity(_find_best_match_location(result))
    assert_allclose(match_location.value, np.array(result.shape) / 2.0 - 0.5 + aia171_test_shift, rtol=1e-3, atol=0)


def test_lower_clip(aia171_test_clipping):
    # No element is less than zero
    test_array = np.asarray([1.1, 0.1, 3.0])
    assert _lower_clip(test_array) == 0
    assert _lower_clip(aia171_test_clipping) == 2.0


def test_upper_clip(aia171_test_clipping):
    assert _upper_clip(aia171_test_clipping) == 1.0
    # No element is greater than zero
    test_array = np.asarray([-1.1, -0.1, -3.0])
    assert _upper_clip(test_array) == 0


def test_calculate_clipping(aia171_test_clipping):
    answer = _calculate_clipping(aia171_test_clipping * u.pix, aia171_test_clipping * u.pix)
    assert_array_almost_equal(answer, ([2.0, 1.0] * u.pix, [2.0, 1.0] * u.pix))


def test_clip_edges():
    a = np.zeros(shape=(341, 156))
    yclip = [4, 0] * u.pix
    xclip = [1, 2] * u.pix
    new_a = _clip_edges(a, yclip, xclip)
    assert a.shape[0] - (yclip[0].value + yclip[1].value) == new_a.shape[0]
    assert a.shape[1] - (xclip[0].value + xclip[1].value) == new_a.shape[1]


def test__default_fmap_function():
    assert _default_fmap_function([1, 2, 3]).dtype == np.float64(1).dtype


#
# The following tests test functions that have mapsequences as inputs
#
# Setup the test mapsequences that have displacements
# Pixel displacements have the y-displacement as the first entry


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
def aia171_test_mc(aia171_test_map, aia171_test_map_layer, aia171_test_mc_pixel_displacements):
    # Create a map that has been shifted a known amount.
    d1 = sp_shift(aia171_test_map_layer, aia171_test_mc_pixel_displacements)
    m1 = Map((d1, aia171_test_map.meta))
    # Create the mapsequence
    return Map([aia171_test_map, m1], sequence=True)


def test_calculate_match_template_shift(
    aia171_test_mc,
    aia171_mc_arcsec_displacements,
    aia171_test_map,
    aia171_test_map_layer,
    aia171_test_map_layer_shape,
):
    # Define these local variables to make the code more readable
    ny = aia171_test_map_layer_shape[0]
    nx = aia171_test_map_layer_shape[1]

    # Test to see if the code can recover the displacements.
    test_displacements = calculate_match_template_shift(aia171_test_mc)
    assert_allclose(test_displacements["x"], aia171_mc_arcsec_displacements["x"], rtol=5e-2, atol=0)
    assert_allclose(test_displacements["y"], aia171_mc_arcsec_displacements["y"], rtol=5e-2, atol=0)

    # Test setting the template as a ndarray
    template_ndarray = aia171_test_map_layer[ny // 4 : 3 * ny // 4, nx // 4 : 3 * nx // 4]
    test_displacements = calculate_match_template_shift(aia171_test_mc, template=template_ndarray)
    assert_allclose(test_displacements["x"], aia171_mc_arcsec_displacements["x"], rtol=5e-2, atol=0)
    assert_allclose(test_displacements["y"], aia171_mc_arcsec_displacements["y"], rtol=5e-2, atol=0)

    # Test setting the template as GenericMap
    submap = aia171_test_map.submap([nx / 4, ny / 4] * u.pix, top_right=[3 * nx / 4, 3 * ny / 4] * u.pix)
    test_displacements = calculate_match_template_shift(aia171_test_mc, template=submap)
    assert_allclose(test_displacements["x"], aia171_mc_arcsec_displacements["x"], rtol=5e-2, atol=0)
    assert_allclose(test_displacements["y"], aia171_mc_arcsec_displacements["y"], rtol=5e-2, atol=0)

    # Test setting the template as something other than a ndarray and a
    # GenericMap. This should throw a ValueError.
    with pytest.raises(ValueError, match="Invalid template."):
        calculate_match_template_shift(aia171_test_mc, template="broken")


def test_mapsequence_coalign_by_match_template(aia171_test_mc, aia171_test_map_layer_shape):
    # Define these local variables to make the code more readable
    ny = aia171_test_map_layer_shape[0]
    nx = aia171_test_map_layer_shape[1]

    # Get the calculated test displacements
    test_displacements = calculate_match_template_shift(aia171_test_mc)

    # Test passing in displacements
    test_mc = mapsequence_coalign_by_match_template(aia171_test_mc, shift=test_displacements)

    # Make sure the output is a mapsequence
    assert isinstance(test_mc, MapSequence)

    # Test returning with no clipping.  Output layers should have the same size
    # as the original input layer.
    test_mc = mapsequence_coalign_by_match_template(aia171_test_mc, clip=False)
    assert test_mc[0].data.shape == aia171_test_map_layer_shape
    assert test_mc[1].data.shape == aia171_test_map_layer_shape

    # Test the returned mapsequence using the default - clipping on.
    # All output layers should have the same size
    # which is smaller than the input by a known amount
    test_mc = mapsequence_coalign_by_match_template(aia171_test_mc)
    x_displacement_pixels = test_displacements["x"] / test_mc[0].scale[0]
    y_displacement_pixels = test_displacements["y"] / test_mc[0].scale[1]
    expected_clipping = _calculate_clipping(y_displacement_pixels, x_displacement_pixels)
    number_of_pixels_clipped = [np.sum(np.abs(expected_clipping[0])), np.sum(np.abs(expected_clipping[1]))]

    assert test_mc[0].data.shape == (
        ny - number_of_pixels_clipped[0].value,
        nx - number_of_pixels_clipped[1].value,
    )
    assert test_mc[1].data.shape == (
        ny - number_of_pixels_clipped[0].value,
        nx - number_of_pixels_clipped[1].value,
    )

    # Test the returned mapsequence explicitly using clip=True.
    # All output layers should have the same size
    # which is smaller than the input by a known amount
    test_mc = mapsequence_coalign_by_match_template(aia171_test_mc, clip=True)
    x_displacement_pixels = test_displacements["x"] / test_mc[0].scale[0]
    y_displacement_pixels = test_displacements["y"] / test_mc[0].scale[1]
    expected_clipping = _calculate_clipping(y_displacement_pixels, x_displacement_pixels)
    number_of_pixels_clipped = [np.sum(np.abs(expected_clipping[0])), np.sum(np.abs(expected_clipping[1]))]

    assert test_mc[0].data.shape == (
        ny - number_of_pixels_clipped[0].value,
        nx - number_of_pixels_clipped[1].value,
    )
    assert test_mc[1].data.shape == (
        ny - number_of_pixels_clipped[0].value,
        nx - number_of_pixels_clipped[1].value,
    )

    # Test that the reference pixel of each map in the coaligned mapsequence is
    # correct.
    for im, m in enumerate(aia171_test_mc):
        for i_s, s in enumerate(["x", "y"]):
            assert_allclose(
                aia171_test_mc[im].reference_pixel[i_s] - test_mc[im].reference_pixel[i_s],
                test_displacements[s][im] / m.scale[i_s],
                rtol=5e-2,
                atol=0,
            )


def test_apply_shifts(aia171_test_map):
    # take two copies of the AIA image and create a test mapsequence.
    mc = Map([aia171_test_map, aia171_test_map], sequence=True)

    # Pixel displacements have the y-displacement as the first entry
    numerical_displacements = {"x": np.asarray([0.0, -2.7]), "y": np.asarray([0.0, -10.4])}
    astropy_displacements = {
        "x": numerical_displacements["x"] * u.pix,
        "y": numerical_displacements["y"] * u.pix,
    }

    # Test to see if the code can detect the fact that the input shifts are not
    # astropy quantities
    with pytest.raises(TypeError):
        apply_shifts(mc, numerical_displacements["y"], astropy_displacements["x"])
    with pytest.raises(TypeError):
        apply_shifts(mc, astropy_displacements["y"], numerical_displacements["x"])
    with pytest.raises(TypeError):
        apply_shifts(mc, numerical_displacements["y"], numerical_displacements["x"])

    # Test returning with no extra options - the code returns a mapsequence only
    test_output = apply_shifts(mc, astropy_displacements["y"], astropy_displacements["x"])
    assert isinstance(test_output, MapSequence)

    # Test returning with no clipping.  Output layers should have the same size
    # as the original input layer.
    test_mc = apply_shifts(mc, astropy_displacements["y"], astropy_displacements["x"], clip=False)
    assert test_mc[0].data.shape == aia171_test_map.data.shape
    assert test_mc[1].data.shape == aia171_test_map.data.shape

    # Test returning with clipping.  Output layers should be smaller than the
    # original layer by a known amount.
    test_mc = apply_shifts(mc, astropy_displacements["y"], astropy_displacements["x"], clip=True)
    for i in range(len(test_mc.maps)):
        clipped = _calculate_clipping(astropy_displacements["y"], astropy_displacements["x"])
        assert test_mc[i].data.shape[0] == mc[i].data.shape[0] - np.max(clipped[0].value)
        assert test_mc[i].data.shape[1] == mc[i].data.shape[1] - np.max(clipped[1].value)

    # Test returning with default clipping.  The default clipping is set to
    # true, that is the mapsequence is clipped.  Output layers should be smaller
    # than the original layer by a known amount.
    test_mc = apply_shifts(mc, astropy_displacements["y"], astropy_displacements["x"])
    for i in range(len(test_mc.maps)):
        clipped = _calculate_clipping(astropy_displacements["y"], astropy_displacements["x"])
        assert test_mc[i].data.shape[0] == mc[i].data.shape[0] - np.max(clipped[0].value)
        assert test_mc[i].data.shape[1] == mc[i].data.shape[1] - np.max(clipped[1].value)

    # Test that keywords are correctly passed
    # Test for an individual keyword
    test_mc = apply_shifts(mc, astropy_displacements["y"], astropy_displacements["x"], clip=False, cval=np.nan)
    assert np.all(np.logical_not(np.isfinite(test_mc[1].data[:, -1])))

    # Test for a combination of keywords, and that changing the interpolation
    # order and how the edges are treated changes the results.
    test_mc1 = apply_shifts(
        mc,
        astropy_displacements["y"],
        astropy_displacements["x"],
        clip=False,
        order=2,
        mode="reflect",
    )
    test_mc2 = apply_shifts(mc, astropy_displacements["y"], astropy_displacements["x"], clip=False)
    assert np.all(test_mc1[1].data[:, -1] != test_mc2[1].data[:, -1])


@pytest.fixture()
def aia171_test_submap(aia171_test_map):
    return aia171_test_map.submap(SkyCoord(((0, 0), (400, 500)) * u.arcsec, frame=aia171_test_map.coordinate_frame))


@pytest.fixture()
def aia171_test_mapsequence(aia171_test_submap):
    m2header = deepcopy(aia171_test_submap.meta)
    m2header["date-obs"] = "2011-02-15T01:00:00.34"
    m2 = sunpy.map.Map((aia171_test_submap.data, m2header))
    m3header = deepcopy(aia171_test_submap.meta)
    m3header["date-obs"] = "2011-02-15T02:00:00.34"
    m3 = sunpy.map.Map((aia171_test_submap.data, m3header))
    return sunpy.map.Map([aia171_test_submap, m2, m3], sequence=True)


@pytest.fixture()
def known_displacements_layer_index0():
    # Known displacements for these mapsequence layers when the layer index is set to 0
    return {"x": np.asarray([0.0, -9.827465, -19.676442]), "y": np.asarray([0.0, 0.251137, 0.490014])}


@pytest.fixture()
def known_displacements_layer_index1():
    # Known displacements for these mapsequence layers when the layer index is set to 1
    return {"x": np.asarray([9.804878, 0.0, -9.827465]), "y": np.asarray([-0.263369, 0.0, 0.251137])}
