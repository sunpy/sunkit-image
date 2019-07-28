import numpy as np
import pytest
import os

import sunkit_image.flct._pyflct as pyflct


@pytest.fixture
def arrays_test():

    a = np.zeros((4, 4))
    b = np.ones((4, 4))
    c = np.zeros((4, 4))

    return (a, b, c)


def test_two_read_write(arrays_test):

    file_name = "temp.dat"

    pyflct.write_two_images(file_name, arrays_test[0], arrays_test[1])

    ier, arr, barr = pyflct.read_two_images(file_name)

    os.remove(file_name)

    assert np.allclose(ier, 1)
    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))


def test_three_read_write(arrays_test):

    file_name = "temp.dat"

    pyflct.write_three_images(file_name, arrays_test[0], arrays_test[1], arrays_test[2])

    ier, arr, barr, carr = pyflct.read_three_images(file_name)

    os.remove(file_name)

    assert np.allclose(ier, 1)
    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))
    assert np.allclose(carr, np.zeros((4, 4)))


def test_swaps(arrays_test):

    result_a, result_b = pyflct.swap_order_two(arrays_test[0], arrays_test[1])

    assert np.allclose(result_a, arrays_test[0])
    assert np.allclose(result_b, arrays_test[1])

    result_a, result_b, result_c = pyflct.swap_order_three(arrays_test[0], arrays_test[1], arrays_test[2])

    assert np.allclose(result_a, arrays_test[0])
    assert np.allclose(result_b, arrays_test[1])
    assert np.allclose(result_c, arrays_test[2])
