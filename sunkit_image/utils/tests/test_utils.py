import numpy as np
import pytest

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from sunkit_image.utils import (
    bandpass_filter,
    bin_edge_summary,
    equally_spaced_bins,
    find_pixel_radii,
    get_radial_intensity_summary,
    erase_loop_in_residual,
    curvature_radius,
    initial_direction_finding,
    loop_add,
    smooth,
)


@pytest.fixture
@pytest.mark.remote_data
def smap():
    import sunpy.data.sample
    from sunpy.data.sample import AIA_171_IMAGE

    return sunpy.map.Map(AIA_171_IMAGE)


def test_equally_spaced_bins():
    # test the default
    esb = equally_spaced_bins()
    assert esb.shape == (2, 100)
    assert esb[0, 0] == 1.0
    assert esb[1, 0] == 1.01
    assert esb[0, 99] == 1.99
    assert esb[1, 99] == 2.00

    # Bins are 0.015 wide
    esb2 = equally_spaced_bins(inner_value=0.5)
    assert esb2.shape == (2, 100)
    assert esb2[0, 0] == 0.5
    assert esb2[1, 0] == 0.515
    assert esb2[0, 99] == 1.985
    assert esb2[1, 99] == 2.00

    # Bins are 0.2 wide
    esb2 = equally_spaced_bins(outer_value=3.0)
    assert esb2.shape == (2, 100)
    assert esb2[0, 0] == 1.0
    assert esb2[1, 0] == 1.02
    assert esb2[0, 99] == 2.98
    assert esb2[1, 99] == 3.00

    # Bins are 0.01 wide
    esb2 = equally_spaced_bins(nbins=1000)
    assert esb2.shape == (2, 1000)
    assert esb2[0, 0] == 1.0
    assert esb2[1, 0] == 1.001
    assert esb2[0, 999] == 1.999
    assert esb2[1, 999] == 2.000

    # The radii have the correct relative sizes
    with pytest.raises(ValueError):
        equally_spaced_bins(inner_value=1.0, outer_value=1.0)
    with pytest.raises(ValueError):
        equally_spaced_bins(inner_value=1.5, outer_value=1.0)

    # The number of bins is strictly greater than 0
    with pytest.raises(ValueError):
        equally_spaced_bins(nbins=0)


def test_bin_edge_summary():
    esb = equally_spaced_bins()

    center = bin_edge_summary(esb, "center")
    assert center.shape == (100,)
    assert center[0] == 1.005
    assert center[99] == 1.995

    left = bin_edge_summary(esb, "left")
    assert left.shape == (100,)
    assert left[0] == 1.0
    assert left[99] == 1.99

    right = bin_edge_summary(esb, "right")
    assert right.shape == (100,)
    assert right[0] == 1.01
    assert right[99] == 2.0

    # Correct selection of summary type
    with pytest.raises(ValueError):
        bin_edge_summary(esb, "should raise the error")

    # The correct shape of bin edges are passed in
    with pytest.raises(ValueError):
        bin_edge_summary(np.arange(0, 10), "center")
    with pytest.raises(ValueError):
        bin_edge_summary(np.zeros((3, 4)), "center")


@pytest.mark.remote_data
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
    assert_quantity_allclose((np.max(pixel_radii)).value, known_maximum_pixel_radius)

    # Test that the new scale is used
    pixel_radii = find_pixel_radii(smap, scale=2 * smap.rsun_obs)
    assert_quantity_allclose(np.max(pixel_radii).value, known_maximum_pixel_radius / 2)


def test_get_radial_intensity_summary():
    # TODO: Write some tests.
    pass


@pytest.fixture
def test_map():
    map_test = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 5.0, 5.0, 1.0],
        [1.0, 5.0, 5.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    return np.array(map_test)


@pytest.fixture
def image():
    return np.ones((4, 4), dtype=np.float32)


def test_bandpass_filter(image, test_map):

    expect = np.zeros((4, 4))
    result = bandpass_filter(image)

    assert np.allclose(expect, result)

    expect = np.array([[0., 0., 0., 0.],
                        [0., 2.22222222, 2.22222222, 0.],
                        [0., 2.22222222, 2.22222222, 0.],
                        [0., 0., 0., 0.]])

    result = bandpass_filter(test_map)

    assert np.allclose(expect, result)

    with pytest.raises(ValueError) as record:
        _ = bandpass_filter(image, 5, 1)

    assert str(record.value) == "nsm1 should be less than nsm2"


def test_smooth(image, test_map):

    filtered = smooth(image, 1)
    assert np.allclose(filtered, image)

    filtered = smooth(image, 4)
    assert np.allclose(filtered, image)

    filtered = smooth(test_map, 1)
    assert np.allclose(filtered, test_map)

    filtered = smooth(test_map, 3)
    expect = np.array([[1., 1., 1., 1.],
                        [1., 2.77777777, 2.77777777, 1.],
                        [1., 2.77777777, 2.77777777, 1.],
                        [1., 1., 1., 1.]])
    
    assert np.allclose(filtered, expect)


def test_erase_loop_in_residual(image, test_map):

    # The starting point of a dummy loop
    istart = 0
    jstart = 1
    width = 1

    # The coordinates of the dummy loop
    xloop = [1, 2, 3]
    yloop = [1, 1, 1]

    result = erase_loop_in_residual(image, istart, jstart, width, xloop, yloop)

    expect = np.array([[0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.]])

    assert np.allclose(expect, result)

    result = erase_loop_in_residual(test_map, istart, jstart, width, xloop, yloop)

    expect = np.array([[0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.],
                       [0., 0., 0., 1.]])

    assert np.allclose(expect, result)


@pytest.fixture
def test_image():

    # An image containing a loop in a straight line
    ima = np.zeros((3, 3), dtype=np.float32)
    ima[0, 1] = 5
    ima[1, 1] = 3
    ima[2, 1] = 0
    return ima


def test_initial_direction_finding(test_image):
    
    # The starting point of the loop i.e. the maximumflux position
    xstart = 0
    ystart = 1
    nlen = 30
    
    # The angle returned is with respect to the ``x`` axis.
    al = initial_direction_finding(test_image, xstart, ystart, nlen)

    # The angle returned is zero because the image has loop in the ``y`` direction but the function
    # assumes the image is transposed so it takes the straight line in the ``x`` direction.
    assert np.allclose(al, 0.0)


def test_curvature_radius(test_image):

    xl = np.zeros((3), dtype=np.float32)
    yl = np.zeros((3), dtype=np.float32)
    zl = np.zeros((3), dtype=np.float32)
    al = np.zeros((3), dtype=np.float32)
    ir = np.zeros((3), dtype=np.float32)

    xl[0] = 0
    yl[0] = 1
    zl[0] = 5
    al[0] = 0.0

    # Using the similar settings in as in the IDL tutorial.
    # This is forward tracing where the first point is after the starting point is being traced.
    xl, yl, zl, al = curvature_radius(test_image, 30, xl, yl, zl, al, ir, 0, 30, 0)

    assert np.allclose(np.ceil(xl[1]), 1)
    assert np.allclose(np.ceil(yl[1]), 1)
    assert np.allclose(zl[1], 3)

    # This is forward tracing where the second point is after the starting point is being traced.
    xl, yl, zl, al = curvature_radius(test_image, 30, xl, yl, zl, al, ir, 1, 30, 0)

    assert np.allclose(np.ceil(xl[2]), 2)
    assert np.allclose(np.ceil(yl[2]), 1)
    assert np.allclose(zl[2], 0)


@pytest.fixture
def parameters_add_loop():

    # Here we are creating dummy coordinates and flux for a loop
    xloop = np.ones(8, dtype=np.float32) * 7
    yloop = np.arange(11,3,-1, dtype=np.float32)
    zloop = np.array([1, 2, 4, 3, 4, 12, 6, 3], dtype=np.float32)

    iloop = 0
    np1 = len(xloop)

    # Calculate the length of each point
    lengths = np.zeros((np1), dtype=np.float32)

    for ip in range(1, np1):
        lengths[ip] = lengths[ip - 1] + np.sqrt((xloop[ip] - xloop[ip - 1]) ** 2 + (yloop[ip] - yloop[ip - 1]) ** 2)

    # The empty structures in which the first loop is stored
    loops = []
    loopfile = None
    
    return (lengths, xloop, yloop, zloop, iloop, loops, loopfile)


def test_add_loop(parameters_add_loop):

    # We call the add_loop function and the values should be placed in the structures
    loopfile, loops, iloop = loop_add(*parameters_add_loop)

    expect_loopfile = np.array([[ 0.,  7., 11.,  1.,  0.],
                                [ 0.,  7., 10.,  2.,  1.],
                                [ 0.,  7.,  9.,  4.,  2.],
                                [ 0.,  7.,  8.,  3.,  3.],
                                [ 0.,  7.,  7.,  4.,  4.],
                                [ 0.,  7.,  6., 12.,  5.],
                                [ 0.,  7.,  5.,  6.,  6.]])

    expect_loops = [[[7.0, 11.0], [7.0, 10.0], [7.0, 9.0], [7.0, 8.0], [7.0, 7.0], [7.0, 6.0], [7.0, 5.0]]]

    assert np.allclose(loopfile, expect_loopfile)
    assert np.allclose(loops, expect_loops)
    assert np.allclose(iloop, 1)

def test_parameters_add_loop(parameters_add_loop):

    lengths, xloop, yloop, zloop, iloop, loops, loopfile = parameters_add_loop

    assert np.allclose(lengths, np.arange(0, 8))
    assert np.allclose(xloop, np.ones(8) * 7)
    assert np.allclose(yloop, np.arange(11,3,-1))
    assert np.allclose(zloop, np.array([1, 2, 4, 3, 4, 12, 6, 3]))
    assert np.allclose(iloop, 0)
    assert (not loops)
    
    if loopfile is not None:
        assert False 
