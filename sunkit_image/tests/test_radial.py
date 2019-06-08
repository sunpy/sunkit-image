import numpy as np
import pytest
import astropy.units as u

import sunpy
import sunpy.map
import sunpy.data.sample
from sunpy.tests.helpers import figure_test

import sunkit_image.utils.utils as utils
import sunkit_image.radial as rad


@pytest.fixture
def map_test1():
    x = np.linspace(-2, 2, 5)
    grid = np.meshgrid(x, x.T)
    test_data1 = np.sqrt(grid[0]**2 + grid[1]**2)
    test_data1 *= 10
    test_data1 = 28-test_data1
    test_data1 = np.round(test_data1)
    header = {'cunit1': 'arcsec', 'cunit2': 'arcsec'}
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
    header = {'cunit1': 'arcsec', 'cunit2': 'arcsec'}
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


def test_nrgf(map_test1, map_test2, radial_bin_edges):

    result = np.zeros_like(map_test1.data)
    expect = rad.nrgf(map_test1, radial_bin_edges, application_radius=0.001*u.R_sun)

    assert np.allclose(expect.data.shape, map_test1.data.shape)
    assert np.allclose(expect.data, result)

    # Hand calculated
    result = [[0.,  1.,  0., -1., 0.],
              [1.,  0.,  0.,  0., -1.],
              [0.,  0.,  0.,  0.,  0.],
              [1.,  0.,  0.,  0., -1.],
              [0.,  1.,  0., -1., 0.]]

    expect = rad.nrgf(map_test2, radial_bin_edges, application_radius=0.001*u.R_sun)

    assert np.allclose(expect.data.shape, map_test2.data.shape)
    assert np.allclose(expect.data, result)


def test_fnrgf(map_test1, map_test2, radial_bin_edges):

    order = 1
    # TODO : Write few more tests with different order

    # Hand calculated
    result = [[-0.,  96., 128.,  96.,  -0.],
              [96., 224., 288., 224.,  96.],
              [128., 288.,   0., 288., 128.],
              [96., 224., 288., 224.,  96.],
              [-0.,  96., 128.,  96.,  -0.]]
    attenuation_coefficients = set_attenuation_coefficients(order)
    expect = rad.fnrgf(map_test1, radial_bin_edges, order, attenuation_coefficients,
                       application_radius=0.001*u.R_sun, number_angular_segments=4)

    assert np.allclose(expect.data.shape, map_test1.data.shape)
    assert np.allclose(expect.data, result)

    # Hand calculated
    result = [[-0., 128., 128.,  96.,  -0.],
              [128., 224., 288., 224.,  96.],
              [128., 288.,   0., 288., 128.],
              [128., 224., 288., 224.,  96.],
              [-0., 128., 128.,  96.,  -0.]]
    expect = rad.fnrgf(map_test2, radial_bin_edges, order, attenuation_coefficients,
                       application_radius=0.001*u.R_sun, number_angular_segments=4)

    assert np.allclose(expect.data.shape, map_test2.data.shape)
    assert np.allclose(expect.data, result)


@pytest.fixture
@pytest.mark.remote_data
def smap():
    return sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)


@figure_test
@pytest.mark.remote_data
def test_fig_nrgf(smap):

    radial_bin_edges = utils.equally_spaced_bins()
    radial_bin_edges *= u.R_sun
    out = rad.nrgf(smap, radial_bin_edges)

    out.plot()


@figure_test
@pytest.mark.remote_data
def test_fig_fnrgf(smap):

    radial_bin_edges = utils.equally_spaced_bins()
    radial_bin_edges *= u.R_sun

    order = 20
    attenuation_coefficients = set_attenuation_coefficients(order)
    out = rad.fnrgf(smap, radial_bin_edges, order, attenuation_coefficients)

    out.plot()


def test_set_attenuation_coefficients():

    order = 1
    # Hand calculated
    expect = [[1, 0.],
              [1, 0.]]

    result = set_attenuation_coefficients(order)

    assert np.allclose(expect, result)

    order = 3
    # Hand calculated
    expect = [[1, 0.75, 0.5, 0.],
              [1, 0.75, 0.5, 0.]]

    result = set_attenuation_coefficients(order)
