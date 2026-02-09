import numpy as np
import pytest

import astropy.units as u

import sunpy.map
from sunpy.map import GenericMap

from sunkit_image.radial import rhef


@pytest.fixture
def map_test():
    """
    Creates a robust synthetic SunPy Map for testing.
    Includes observer coordinates to prevent SunpyMetadataWarnings.
    """
    data = np.ones((20, 20))
    meta = {
        'ctype1': 'HPLN-TAN', 'cunit1': 'arcsec', 'crval1': 0, 'cdelt1': 1, 'crpix1': 10,
        'ctype2': 'HPLT-TAN', 'cunit2': 'arcsec', 'crval2': 0, 'cdelt2': 1, 'crpix2': 10,
        'telescop': 'SUNPY', 'date-obs': '2023-01-01T00:00:00',
        'dsun_obs': 149597870700.0,
        'hgln_obs': 0.0,
        'hglt_obs': 0.0,
        'rsun_ref': 696000000.0,
        'rsun_obs': 900.0,
    }
    return sunpy.map.Map(data, meta)

def test_rhef_returns_map(map_test):
    # Verify that the default function returns a valid Map object
    result = rhef(map_test)
    assert isinstance(result, GenericMap)
    assert result.data.shape == map_test.data.shape

def test_rhef_upsilon_parameter(map_test):
    # Test that varying upsilon does not crash the function
    for upsilon_val in [0.1, 0.5, 0.9]:
        result = rhef(map_test, upsilon=upsilon_val)
        assert isinstance(result, GenericMap)

def test_rhef_plot_settings(map_test):
    # The RHEF algorithm normalizes data, so plot settings must be reset
    result = rhef(map_test)
    assert result.plot_settings.get("norm") is None

def test_rhef_with_vignette(map_test):
    # Test with the vignette argument (masking outer edges)
    result = rhef(map_test, vignette=1.1 * u.R_sun)
    assert isinstance(result, GenericMap)
