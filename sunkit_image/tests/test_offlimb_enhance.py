import numpy as np
import pytest
import astropy.units as u

import sunpy
import sunpy.map
import sunpy.data.test
import sunkit_image.utils.utils as utils
import sunkit_image.offlimb_enhance as off


@pytest.fixture
def map_test1():
        x = np.linspace(-2, 2, 5)
        grid = np.meshgrid(x, x.T)
        test_data1 = np.sqrt(grid[0]**2 + grid[1]**2)
        test_data1 *= 10
        test_data1 = 28-test_data1
        test_data1 = np.round(test_data1)
        header = {}
        test_map1 = sunpy.map.Map((test_data1, header))
        return test_map1

@pytest.fixture
def map_test2():
        x = np.linspace(-2, 2, 5)
        grid = np.meshgrid(x, x.T)
        test_data1 = np.sqrt(grid[0]**2 + grid[1]**2)
        test_data1 *= 10
        test_data1 = 28-test_data1
        test_data1 = np.round(test_data1)
        header = {}
        test_data2 = np.where(test_data1[:, 0:2] == 6, 8, test_data1[:, 0:2])
        test_data2 = np.concatenate((test_data2, test_data1[:, 2:]), axis=1)
        test_map2 = sunpy.map.Map((test_data2, header))
        return test_map2


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


def test_normalizing_radial_gradient_filter(map_test1, map_test2, radial_bin_edges):

    with pytest.warns(Warning, match="Missing metadata for solar radius: assuming photospheric limb as seen from Earth"):

        result = np.zeros_like(map_test1.data)
        expect = off.normalizing_radial_gradient_filter(map_test1, radial_bin_edges,
                                                        application_radius=0.001*u.R_sun)

        assert np.allclose(expect.data.shape, map_test1.data.shape)
        assert np.allclose(expect.data, result)

        result = [[0.,  1.,  0., -1., 0.],
                  [1.,  0.,  0.,  0., -1.],
                  [0.,  0.,  0.,  0.,  0.],
                  [1.,  0.,  0.,  0., -1.],
                  [0.,  1.,  0., -1., 0.]]
        
        expect = off.normalizing_radial_gradient_filter(map_test2, radial_bin_edges,
                                                        application_radius=0.001*u.R_sun)

        assert np.allclose(expect.data.shape, map_test2.data.shape)
        assert np.allclose(expect.data, result)


# These tests will fail for the time being
@pytest.mark.parametrize("order", [(20), (33)])
def test_fourier_normalizing_radial_gradient_filter(order, map_test1, radial_bin_edges):

    with pytest.warns(Warning, match="Missing metadata for solar radius: assuming photospheric limb as seen from Earth"):

        result = np.zeros_like(map_test1.data)
        attenuation_coefficients = set_attenuation_coefficients(order)
        expect = off.fourier_normalizing_radial_gradient_filter(map_test1, radial_bin_edges, order,
                                                                attenuation_coefficients,
                                                                application_radius=0.001*u.R_sun)
        assert np.allclose(expect.data.shape, map_test1.data.shape)
        assert np.allclose(expect.data, result)
