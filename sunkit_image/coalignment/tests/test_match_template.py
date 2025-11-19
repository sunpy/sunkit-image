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






def test_find_best_match_location(aia171_test_map, aia171_test_template, aia171_test_shift):
    with np.errstate(all='ignore'):
        result = match_template(aia171_test_map.data, aia171_test_template)
    match_location = u.Quantity(_find_best_match_location(result))
    assert_allclose(match_location.value, np.array(result.shape) / 2.0 - 0.5 + aia171_test_shift, rtol=1e-3, atol=0)


def test_registered_method():
    assert REGISTERED_METHODS["match_template"] is not None
