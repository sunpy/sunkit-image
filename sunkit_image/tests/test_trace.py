import matplotlib.pyplot as plt
import astropy
import numpy as np
import pytest
import os

import sunkit_image.trace as trace
import sunkit_image.data.test as data
from sunpy.tests.helpers import figure_test


@pytest.fixture
@pytest.mark.remote_data
def image():

    im = astropy.io.fits.getdata("http://www.lmsal.com/~aschwand/software/tracing/TRACE_19980519.fits", ignore_missing_end=True)
    return im


@pytest.fixture
def filepath_IDL():

    filepath = data.get_test_filepath("IDL.txt")
    return filepath


@pytest.fixture
def filepath_py():

    filepath = data.get_test_filepath("Python.txt")
    return filepath


@pytest.mark.remote_data
def test_occult2_remote(image, filepath_IDL, filepath_py):

    # Testing on the same input files as in the IDL tutorial
    loops = trace.occult2(image, nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0, file=True)

    result = np.loadtxt("loops.txt")

    os.remove("loops.txt")

    # First we will verify on the Python version of the output which we know is correct as we had plotted
    # it earlirer and also have made visual comparisons
    expect = np.loadtxt(filepath_py)
    assert np.allclose(expect, result)
    assert np.allclose(expect[-1, 0], len(loops) - 1)

    x = []
    y = []
    for loop in loops:
        for points in loop:
            x.append(points[0])
            y.append(points[1])

    assert np.allclose(expect[:, 1], x)
    assert np.allclose(expect[:, 2], y)

    # Now we will test on the IDL output data
    # We know that the python code detects one point extra than the IDL code.
    # So to test it we will remove that point
    result = np.delete(result, (1745), axis=0)

    # Reading the IDL file
    expect = np.loadtxt(filepath_IDL)

    # Taking all the coords
    coords_py = result[:, 1:3]
    coords_idl = expect[:, 1:3]

    # Checking all the coordinates must be close to each other
    assert np.allclose(coords_py, coords_idl, atol=1e-0, rtol=1e-10)

    # We devise one more test where we will find the distance between the Python and IDL points
    # For the algorithm to work correctly this distance should be very small
    diff = coords_idl - coords_py
    square_diff = diff ** 2
    sum_diff = np.sum(square_diff, axis=1)

    distance = np.sqrt(sum_diff)
    assert all(distance < 0.11)


@figure_test
@pytest.mark.remote_data
def test_occult2_fig(image):

    # A figure test for occult2, the plot is same as the one in the IDL tutorial
    loops = trace.occult2(image, nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)

    for loop in loops:

        # We collect all the ``x`` and ``y`` coordinates in seperate lists for plotting.
        x = []
        y = []
        for points in loop:
            x.append(points[0])
            y.append(points[1])

        plt.plot(x, y, 'b')


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
    loops = trace.occult2(image_test, nsm1=1, rmin=30, lmin=0, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)

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
    loops = trace.occult2(image_test, nsm1=1, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)
    assert (not loops)

    # This check is used to verify whether the RuntimeError is triggered
    with pytest.raises(RuntimeError) as record:
        _ = trace.occult2(test_image, nsm1=3, rmin=30, lmin=25, nstruc=1000, ngap=0, qthresh1=0.0, qthresh2=3.0)

    assert str(record.value) == "The filter size is very large compared to the size of the image. The entire image zeros out while smoothing the image edges after filtering."
