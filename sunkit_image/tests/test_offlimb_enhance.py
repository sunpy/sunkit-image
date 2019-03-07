import numpy as np
import pytest
import astropy.units as u

import sunpy
import sunpy.map
import sunpy.data.test
import sunkit_image.utils.utils as utils
import sunkit_image.offlimb_enhance as off


test_map_data = np.ones((5, 5))
header = {}
test_map = sunpy.map.Map((test_map_data, header))
radial_bin_edges = utils.equally_spaced_bins()
radial_bin_edges = radial_bin_edges*u.R_sun


def set_attenuation_coefficients(order):
    attenuation_coefficients = np.zeros((2, order + 1))
    attenuation_coefficients[0][:] = np.linspace(1, 0, order + 1)
    attenuation_coefficients[1][:] = np.linspace(1, 0, order + 1)
    return attenuation_coefficients


def test_normalizing_radial_gradient_filter():

    with pytest.warns(Warning, match="Missing metadata for solar radius: assuming photospheric limb as seen from Earth"):

        result = np.zeros_like(test_map_data)
        expect = off.normalizing_radial_gradient_filter(test_map, radial_bin_edges)

        assert np.allclose(expect.data.shape, test_map.data.shape)
        assert np.allclose(expect.data, result)


def test_fourier_normalizing_radial_gradient_filter():

    with pytest.warns(Warning, match="Missing metadata for solar radius: assuming photospheric limb as seen from Earth"):

        result = np.zeros_like(test_map_data)

        order = 20
        attenuation_coefficients = set_attenuation_coefficients(order)
        expect = off.fourier_normalizing_radial_gradient_filter(test_map, radial_bin_edges, order,
                                                                attenuation_coefficients)
        assert np.allclose(expect.data.shape, test_map.data.shape)
        assert np.allclose(expect.data, result)

        order = 33
        attenuation_coefficients = set_attenuation_coefficients(order)
        expect = off.fourier_normalizing_radial_gradient_filter(test_map, radial_bin_edges, order,
                                                                attenuation_coefficients)
        assert np.allclose(expect.data.shape, test_map.data.shape)
        assert np.allclose(expect.data, result)
