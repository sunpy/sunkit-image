import numpy as np
import pytest
from numpy.testing import assert_allclose
from skimage.feature import match_template

import astropy.units as u

from sunkit_image.coalignment.interface import REGISTERED_METHODS
from sunkit_image.coalignment.match_template import (
    _find_best_match_location,
    _get_correlation_shifts,
    _parabolic_turning_point,
)




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
    with pytest.raises(ValueError, match=r"Input array dimension should not be greater than 3 in any dimension."):
        _get_correlation_shifts(test_array)

    test_array = np.zeros((3, 4))
    with pytest.raises(ValueError, match=r"Input array dimension should not be greater than 3 in any dimension."):
        _get_correlation_shifts(test_array)


def test_find_best_match_location(aia171_test_map, aia171_test_template, aia171_test_shift):
    with np.errstate(all='ignore'):
        result = match_template(aia171_test_map.data, aia171_test_template)
    match_location = u.Quantity(_find_best_match_location(result))
    assert_allclose(match_location.value, np.array(result.shape) / 2.0 - 0.5 + aia171_test_shift, rtol=1e-3, atol=0)


def test_registered_method():
    assert REGISTERED_METHODS["match_template"] is not None
