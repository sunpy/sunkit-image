#
# Tests for the utilities
#

from __future__ import absolute_import, division, print_function

import pytest

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord, BaseCoordinateFrame

from sunpy.map import Map
from sunpy.data.sample import AIA_171_IMAGE
from sunkit_image.utils.utils import _equally_spaced_bins, bin_edge_summary, find_pixel_radii, \
    get_radial_intensity_summary, all_pixel_indices_from_map, all_coordinates_from_map, \
    locations_satisfying_condition_relative_to_radius, locations_satisfying_radial_conditions, \
    permitted_comparisons


@pytest.fixture
def smap():
    return Map(AIA_171_IMAGE)


@pytest.fixture
def sub_smap(smap):
    return smap.submap((0, 0)*u.pix, (50, 60)*u.pix)


def test_all_pixel_indices_from_map(sub_smap):
    pixel_indices = all_pixel_indices_from_map(sub_smap)
    shape = sub_smap.data.shape
    ny = shape[0]
    nx = shape[1]
    assert np.all(pixel_indices.shape == (2, ny, nx))
    assert np.all(pixel_indices.unit == u.pix)
    assert np.all(pixel_indices[:, 0, 0] == [0., 0.] * u.pix)
    assert np.all(pixel_indices[:, 0, nx-1] == [nx-1, 0.] * u.pix)
    assert np.all(pixel_indices[:, ny-1, 0] == [0., ny-1] * u.pix)
    assert np.all(pixel_indices[:, ny-1, nx-1] == [nx-1, ny-1] * u.pix)


def test_all_coordinates_from_map(sub_smap):
    coordinates = all_coordinates_from_map(sub_smap)
    shape = sub_smap.data.shape
    assert coordinates.shape == (shape[0], shape[1])
    assert isinstance(coordinates, SkyCoord)
    assert isinstance(coordinates.frame, BaseCoordinateFrame)
    assert coordinates.frame.name == sub_smap.coordinate_frame.name


def test_find_pixel_radii(smap):
    # The known maximum radius
    known_maximum_pixel_radius = 1.84183121

    # Calculate the pixel radii
    pixel_radii = find_pixel_radii(smap)

    # The shape of the pixel radii is the same as the input map
    assert pixel_radii.shape[0] == int(smap.dimensions[0].value)
    assert pixel_radii.shape[1] == int(smap.dimensions[1].value)

    # Make sure the unit is solar radii
    assert pixel_radii.unit == u.R_sun

    # Make sure the maximum
    assert np.allclose(np.max(pixel_radii).value, known_maximum_pixel_radius)

    # Test that the new scale is used
    pixel_radii = find_pixel_radii(smap, scale=2*smap.rsun_obs)
    assert np.allclose(np.max(pixel_radii).value, known_maximum_pixel_radius / 2)


def test_locations_satisfying_condition_relative_to_radius(smap):
    for comparison in permitted_comparisons:
        for scale in (None, 0.5*smap.rsun_obs, 2*smap.rsun_obs):
            for radius in (None, 0.5*u.R_sun, 1.5*u.R_sun):
                lscrtr = locations_satisfying_condition_relative_to_radius(smap,
                                                                           comparison=comparison,
                                                                           scale=scale,
                                                                           radius=radius)
                # Test the shape of the output
                assert lscrtr.shape == smap.data.shape

    # Test the failure to find a permitted comparison operator
    with pytest.raises(ValueError):
        locations_satisfying_condition_relative_to_radius(smap, comparison=np.sin)


def test_locations_satisfying_radial_conditions(smap):
    for comparison1 in permitted_comparisons:
        for comparison2 in permitted_comparisons:
            comparison = (comparison1, comparison2)
            for scale in (None, 0.5*smap.rsun_obs, 2*smap.rsun_obs):
                for radii in (None, (0.6*u.R_sun, 1.5*u.R_sun)):
                    lscrtr = locations_satisfying_radial_conditions(smap,
                                                                    comparison=comparison,
                                                                    scale=scale,
                                                                    radii=radii)
                    # Test the shape of the output
                    assert lscrtr.shape == smap.data.shape

    # Test the failure to find a permitted comparison operator
    with pytest.raises(ValueError):
        locations_satisfying_radial_conditions(smap, comparison=(np.cos, np.sin))


def test_equally_spaced_bins():
    # test the default
    esb = _equally_spaced_bins()
    assert esb.shape == (2, 100)
    assert esb[0, 0] == 1.0
    assert esb[1, 0] == 1.01
    assert esb[0, 99] == 1.99
    assert esb[1, 99] == 2.00

    # Bins are 0.015 wide
    esb2 = _equally_spaced_bins(inner_value=0.5)
    assert esb2.shape == (2, 100)
    assert esb2[0, 0] == 0.5
    assert esb2[1, 0] == 0.515
    assert esb2[0, 99] == 1.985
    assert esb2[1, 99] == 2.00

    # Bins are 0.2 wide
    esb2 = _equally_spaced_bins(outer_value=3.0)
    assert esb2.shape == (2, 100)
    assert esb2[0, 0] == 1.0
    assert esb2[1, 0] == 1.02
    assert esb2[0, 99] == 2.98
    assert esb2[1, 99] == 3.00

    # Bins are 0.01 wide
    esb2 = _equally_spaced_bins(nbins=1000)
    assert esb2.shape == (2, 1000)
    assert esb2[0, 0] == 1.0
    assert esb2[1, 0] == 1.001
    assert esb2[0, 999] == 1.999
    assert esb2[1, 999] == 2.000

    # The radii have the correct relative sizes
    with pytest.raises(ValueError):
        _equally_spaced_bins(inner_value=1.0, outer_value=1.0)
    with pytest.raises(ValueError):
        _equally_spaced_bins(inner_value=1.5, outer_value=1.0)

    # The number of bins is strictly greater than 0
    with pytest.raises(ValueError):
        _equally_spaced_bins(nbins=0)


def test_bin_edge_summary():
    esb = _equally_spaced_bins()

    center = bin_edge_summary(esb, 'center')
    assert center.shape == (100,)
    assert center[0] == 1.005
    assert center[99] == 1.995

    left = bin_edge_summary(esb, 'left')
    assert left.shape == (100,)
    assert left[0] == 1.0
    assert left[99] == 1.99

    right = bin_edge_summary(esb, 'right')
    assert right.shape == (100,)
    assert right[0] == 1.01
    assert right[99] == 2.0

    # Correct selection of summary type
    with pytest.raises(ValueError):
        bin_edge_summary(esb, 'should raise the error')

    # The correct shape of bin edges are passed in
    with pytest.raises(ValueError):
        bin_edge_summary(np.arange(0, 10), 'center')
    with pytest.raises(ValueError):
        bin_edge_summary(np.zeros((3, 4)), 'center')


def test_get_radial_intensity_summary():
    pass

