import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from skimage.feature import match_template

from sunkit_image.coalignment_module.match_template import (
    _calculate_clipping,
    _clip_edges,
    _find_best_match_location,
    _get_correlation_shifts,
    _lower_clip,
    _parabolic_turning_point,
    _upper_clip,
)


def test_parabolic_turning_point():
    assert _parabolic_turning_point(np.asarray([6.0, 2.0, 0.0])) == 1.5


def test_get_correlation_shifts():
    # Input array is 3 by 3, the most common case
    test_array = np.zeros((3, 3))
    test_array[1, 1] = 1
    test_array[2, 1] = 0.6
    test_array[1, 2] = 0.2
    y_test, x_test = _get_correlation_shifts(test_array)
    assert_allclose(y_test, 0.214285714286, rtol=1e-2, atol=0)
    assert_allclose(x_test, 0.0555555555556, rtol=1e-2, atol=0)

    # Input array is smaller in one direction than the other.
    test_array = np.zeros((2, 2))
    test_array[0, 0] = 0.1
    test_array[0, 1] = 0.2
    test_array[1, 0] = 0.4
    test_array[1, 1] = 0.3
    y_test, x_test = _get_correlation_shifts(test_array)
    assert_allclose(y_test, 1.0, rtol=1e-2, atol=0)
    assert_allclose(x_test, 0.0, rtol=1e-2, atol=0)

    # Input array is too big in either direction
    test_array = np.zeros((4, 3))
    with pytest.raises(ValueError, match="Input array dimension should not be greater than 3 in any dimension."):
        _get_correlation_shifts(test_array)

    test_array = np.zeros((3, 4))
    with pytest.raises(ValueError, match="Input array dimension should not be greater than 3 in any dimension."):
        _get_correlation_shifts(test_array)


def test_find_best_match_location(aia171_test_map_layer, aia171_test_template, aia171_test_shift):
    result = match_template(aia171_test_map_layer, aia171_test_template)
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
