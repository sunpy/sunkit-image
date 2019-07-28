import numpy as np
import pytest

from sunpy.tests.helpers import skip_windows

import sunkit_image.data.test as data
from sunkit_image.flct import flct
import sunkit_image.flct._pyflct as pyflct


@pytest.fixture
def images():

    filepath1 = data.get_test_filepath("hashgauss_1.csv")
    filepath2 = data.get_test_filepath("hashgauss_2.csv")

    # These CSV files were created using the IDL IO routines but their order is
    # is not swapped here because we will do it in the flct function.
    image1 = np.genfromtxt(filepath1, delimiter=",")
    image2 = np.genfromtxt(filepath2, delimiter=",")

    return (image1, image2)


@pytest.fixture
def outputs():

    filepath_x = data.get_test_filepath("testgauss_vx.csv")
    filepath_y = data.get_test_filepath("testgauss_vy.csv")
    filepath_m = data.get_test_filepath("testgauss_vm.csv")

    expect_x = np.genfromtxt(filepath_x, delimiter=",")
    expect_y = np.genfromtxt(filepath_y, delimiter=",")
    expect_m = np.genfromtxt(filepath_m, delimiter=",")

    # Since these CSV files were created by reading the dat file on FLCT website
    # their order needs to be rectified.
    expect_x, expect_y, expect_m = pyflct.swap_order_three(expect_x, expect_y, expect_m)

    return (expect_x, expect_y, expect_m)


@skip_windows
def test_pyflct(images, outputs):

    vx, vy, vm = flct(images[0], images[1], "column", 1, 1, 5, kr=0.5)

    assert np.allclose(vx, outputs[0])
    assert np.allclose(vy, outputs[1])
    assert np.allclose(vm, outputs[2])
