import numpy as np
import pytest
import astropy.units as u

import sunpy
import sunpy.map
import sunpy.data.test
import sunkit_image.utils.utils as utils
import sunkit_image.offlimb_enhance as off


@pytest.fixture
def map_test():
        x = np.linspace(-2, 2, 5)
        grid = np.meshgrid(x, x.T)
        test_data = np.sqrt(grid[0]**2 + grid[1]**2)
        test_data *= 10
        test_data = 28-test_data
        test_data = np.round(test_data)
        header = {}
        test_map = sunpy.map.Map((test_data, header))
        return test_map


@pytest.fixture
def radial_bin_edges():
        radial_bins = utils.equally_spaced_bins(inner_value=0.001, outer_value=0.003, nbins=5)
        radial_bins = radial_bins * u.R_sun
        return radial_bins


def set_attenuation_coefficients(order):
    attenuation_coefficients = np.zeros((2, order + 1))
    attenuation_coefficients[0][:] = np.linspace(1, 0, order + 1)
    attenuation_coefficients[1][:] = np.linspace(1, 0, order + 1)
    return attenuation_coefficients


def test_normalizing_radial_gradient_filter(map_test, radial_bin_edges):

    with pytest.warns(Warning, match="Missing metadata for solar radius: assuming photospheric limb as seen from Earth"):

        result = np.zeros_like(map_test.data)
        expect = off.normalizing_radial_gradient_filter(map_test, radial_bin_edges,
                                                        application_radius=0.001*u.R_sun)

        assert np.allclose(expect.data.shape, map_test.data.shape)
        assert np.allclose(expect.data, result)


## These tests will fail for the time being
@pytest.mark.parametrize("order", [(20), (33)])
def test_fourier_normalizing_radial_gradient_filter(order, map_test, radial_bin_edges):

    with pytest.warns(Warning, match="Missing metadata for solar radius: assuming photospheric limb as seen from Earth"):

        result = np.zeros_like(map_test.data)
        attenuation_coefficients = set_attenuation_coefficients(order)
        expect = off.fourier_normalizing_radial_gradient_filter(map_test, radial_bin_edges, order,
                                                                attenuation_coefficients,
                                                                application_radius=0.001*u.R_sun)
        assert np.allclose(expect.data.shape, map_test.data.shape)
        assert np.allclose(expect.data, result)
