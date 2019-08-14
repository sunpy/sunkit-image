import matplotlib.pyplot as plt
import astropy
import numpy as np
import pytest
import os

import sunkit_image.trace as trace
import sunkit_image.data.test as data


@pytest.fixture
@pytest.mark.remote_data
def image():

    im = astropy.io.fits.getdata("http://www.lmsal.com/~aschwand/software/tracing/TRACE_19980519.fits", ignore_missing_end=True)
    return im

@pytest.fixture
def filepath():

    filepath_IDL = data.get_test_filepath("IDL.txt")
    return filepath_IDL


@pytest.mark.remote_data
def test_occult2(image, filepath):

    loops = trace.occult2(image, nsm1=3, rmin=30, lmin=25, nstruc=1000, nloop=1000, ngap=0, qthresh1=0.0, qthresh2=3.0, file=True)
    
    expect = np.loadtxt(filepath)
    result = np.loadtxt("loops.txt")

    os.remove("loops.txt")

    assert np.allclose(expect, result)
    assert np.allclose(expect[-1, 0], len(loops) - 1)

    x = []
    y = []
    for loop in loops:
        for points in loop:
            x.append(points[0])
            y.append(points[1])

    assert np.allclose(expect[:,1], x)
    assert np.allclose(expect[:,2], y)
