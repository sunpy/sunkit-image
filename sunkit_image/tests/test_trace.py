import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from astropy.io import fits

import sunkit_image.data.test as data
from sunkit_image.tests.helpers import figure_test
from sunkit_image.trace import (
    bandpass_filter,
    curvature_radius,
    erase_loop_in_image,
    initial_direction_finding,
    loop_add,
    occult2,
    smooth,
)


@pytest.fixture
@pytest.mark.remote_data
def image_remote():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)
        im = fits.getdata("http://data.sunpy.org/sunkit-image/trace_1998-05-19T22:21:43.000_171_1024.fits")
        return im


@pytest.fixture
def filepath_IDL():

    filepath = data.get_test_filepath("IDL.txt")
    return filepath


@pytest.mark.remote_data
def test_occult2_remote(image_remote, filepath_IDL):

    # Testing on the same input files as in the IDL tutorial
    loops = occult2(image_remote, nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)

    # Taking all the x and y coordinates in seperate lists
    x = []
    y = []
    for loop in loops:
        for points in loop:
            x.append(points[0])
            y.append(points[1])

    # Creating a numpy array of all the loop points for ease of comparison
    X = np.array(x)
    Y = np.array(y)
    coords_py = np.c_[X, Y]

    # Now we will test on the IDL output data

    # Reading the IDL file
    expect = np.loadtxt(filepath_IDL)

    # Validating the number of loops
    assert np.allclose(expect[-1, 0] + 1, len(loops))

    # We know that the python code detects one point extra than the IDL code.
    # So to test it we will remove that point.
    coords_py = np.delete(coords_py, (1745), axis=0)

    # Taking all the coords from the IDL form
    coords_idl = expect[:, 1:3]

    # Checking all the coordinates must be close to each other
    assert np.allclose(coords_py, coords_idl, atol=1e-0, rtol=1e-10)

    # We devise one more test where we will find the distance between the Python and IDL points
    # For the algorithm to work correctly this distance should be very small.
    diff = coords_idl - coords_py
    square_diff = diff**2
    sum_diff = np.sum(square_diff, axis=1)

    distance = np.sqrt(sum_diff)

    # The maximum distance between the IDL points and the Python points was found to be 0.11 pixels.
    assert all(distance < 0.11)


@figure_test
@pytest.mark.remote_data
def test_occult2_fig(image_remote):

    # A figure test for occult2, the plot is same as the one in the IDL tutorial
    loops = occult2(image_remote, nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)

    for loop in loops:

        # We collect all the ``x`` and ``y`` coordinates in seperate lists for plotting.
        x = []
        y = []
        for points in loop:
            x.append(points[0])
            y.append(points[1])

        plt.plot(x, y, "b")


@pytest.fixture
def test_image():

    # An image containing a loop in a straight line
    ima = np.zeros((3, 3), dtype=np.float32)
    ima[0, 1] = 5
    ima[1, 1] = 3
    ima[2, 1] = 0
    return ima


@pytest.fixture
def image_test():

    # An image containing a loop in a straight line
    ima = np.zeros((15, 15), dtype=np.float32)
    ima[:, 7] = 1
    ima[3:12, 7] = [4, 3, 6, 12, 4, 3, 4, 2, 1]

    return ima


def test_occult2(test_image, image_test):

    # Set of checks which does not require remote data

    # The first test were valid loops are detected
    loops = occult2(image_test, nsm1=1, rmin=30, lmin=0, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)

    for loop in loops:

        # We collect all the ``x`` and ``y`` coordinates in seperate lists
        x = []
        y = []
        for points in loop:
            x.append(points[0])
            y.append(points[1])

    # From the input image it is clear that all x coordinate is 7.
    assert np.allclose(np.round(x), np.ones(8) * 7)
    # All the y coords are [11, 10, ..., 4]
    assert np.allclose(np.round(y), np.arange(11, 3, -1))

    # This check will return an empty list as no loop is detected
    loops = occult2(image_test, nsm1=1, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)
    assert not loops

    # This check is used to verify whether the RuntimeError is triggered
    with pytest.raises(RuntimeError) as record:
        _ = occult2(test_image, nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)

    assert (
        str(record.value)
        == "The filter size is very large compared to the size of the image."
        + " The entire image zeros out while smoothing the image edges after filtering."
    )


@pytest.fixture
def test_map():
    map_test = [[1.0, 1.0, 1.0, 1.0], [1.0, 5.0, 5.0, 1.0], [1.0, 5.0, 5.0, 1.0], [1.0, 1.0, 1.0, 1.0]]

    return np.array(map_test)


@pytest.fixture
def image():
    return np.ones((4, 4), dtype=np.float32)


def test_bandpass_filter(image, test_map):

    expect = np.zeros((4, 4))
    result = bandpass_filter(image)

    assert np.allclose(expect, result)

    expect = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 2.22222222, 2.22222222, 0.0],
            [0.0, 2.22222222, 2.22222222, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

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
    expect = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.77777777, 2.77777777, 1.0],
            [1.0, 2.77777777, 2.77777777, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    assert np.allclose(filtered, expect)


def test_erase_loop_in_image(image, test_map):

    # The starting point of a dummy loop
    istart = 0
    jstart = 1
    width = 1

    # The coordinates of the dummy loop
    xloop = [1, 2, 3]
    yloop = [1, 1, 1]

    result = erase_loop_in_image(image, istart, jstart, width, xloop, yloop)

    expect = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

    assert np.allclose(expect, result)

    result = erase_loop_in_image(test_map, istart, jstart, width, xloop, yloop)

    expect = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])

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
    yloop = np.arange(11, 3, -1, dtype=np.float32)
    zloop = np.array([1, 2, 4, 3, 4, 12, 6, 3], dtype=np.float32)

    iloop = 0
    np1 = len(xloop)

    # Calculate the length of each point
    lengths = np.zeros((np1), dtype=np.float32)

    for ip in range(1, np1):
        lengths[ip] = lengths[ip - 1] + np.sqrt((xloop[ip] - xloop[ip - 1]) ** 2 + (yloop[ip] - yloop[ip - 1]) ** 2)

    # The empty structures in which the first loop is stored
    loops = []

    return (lengths, xloop, yloop, zloop, iloop, loops)


def test_add_loop(parameters_add_loop):

    # We call the add_loop function and the values should be placed in the structures
    loops, iloop = loop_add(*parameters_add_loop)

    expect_loops = [[[7.0, 11.0], [7.0, 10.0], [7.0, 9.0], [7.0, 8.0], [7.0, 7.0], [7.0, 6.0], [7.0, 5.0]]]

    assert np.allclose(loops, expect_loops)
    assert np.allclose(iloop, 1)


def test_parameters_add_loop(parameters_add_loop):

    lengths, xloop, yloop, zloop, iloop, loops = parameters_add_loop

    assert np.allclose(lengths, np.arange(0, 8))
    assert np.allclose(xloop, np.ones(8) * 7)
    assert np.allclose(yloop, np.arange(11, 3, -1))
    assert np.allclose(zloop, np.array([1, 2, 4, 3, 4, 12, 6, 3]))
    assert np.allclose(iloop, 0)
    assert not loops
