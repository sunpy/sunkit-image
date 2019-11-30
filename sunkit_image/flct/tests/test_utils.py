import os
import sys

import numpy as np
import pytest

from sunkit_image.flct.utils import *

# We skip this file as the extension is not built on windows.
if sys.platform.startswith("win"):
    pytest.skip("Tests will not run on windows", allow_module_level=True)


# Testing the FLCT subroutines
@pytest.fixture
def arrays_test():

    a = np.zeros((4, 4))
    b = np.ones((4, 4))
    c = np.zeros((4, 4))

    return (a, b, c)


def test_two_read_write(arrays_test):

    """
    This test is written to veify that the wrapped function is able to
    correctly write and read two numpy arrays.
    """

    # Here a temporary dat file is created and two numpy arrays are written in it.
    file_name = "temp.dat"

    write_2_images(file_name, arrays_test[0], arrays_test[1])

    # The temporary dat file is then read again using the wrapped function
    arr, barr = read_2_images(file_name)

    # The temporary file is then deleted
    os.remove(file_name)

    # Then it is verified whether the arrays that were actually written,
    # the same arrays are read back.
    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))

    # The same thing as above is repeated the only difference being that the arrays are
    # both written and read back in column major order
    write_2_images(file_name, arrays_test[0], arrays_test[1], order="column")

    arr, barr = read_2_images(file_name, order="column")

    os.remove(file_name)

    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))

    # The below series of checks below are just to check that the ValueErrors are triggered
    # when wrong value of order is given to any read or write function.
    order = "random"

    with pytest.raises(ValueError) as record:
        write_2_images(file_name, arrays_test[0], arrays_test[1], order)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )

    order = "random"

    with pytest.raises(ValueError) as record:
        _ = read_2_images(file_name, order)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )


def test_three_read_write(arrays_test):

    """
    This test is written to verify that the wrapped function is able to
    correctly write and read three numpy arrays.
    """

    # Here a temporary dat file is created and three numpy arrays are written in it.
    file_name = "temp.dat"

    write_3_images(file_name, arrays_test[0], arrays_test[1], arrays_test[2])

    # The temporary dat file is then read again using the wrapped function
    arr, barr, carr = read_3_images(file_name)

    # The temporary file is then deleted
    os.remove(file_name)

    # Then it is verified whether the arrays that were actually written, the same arrays
    # are read back.
    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))
    assert np.allclose(carr, np.zeros((4, 4)))

    # The same thing as above is repeated the only difference being that the arrays
    # are both written and read back in column major order
    write_3_images(file_name, arrays_test[0], arrays_test[1], arrays_test[2], order="column")

    arr, barr, carr = read_3_images(file_name, order="column")

    os.remove(file_name)

    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))
    assert np.allclose(carr, np.zeros((4, 4)))

    # The below series of checks below are just to check that the ValueErrors are triggered
    # when wrong value of order is given to any read or write function.
    order = "random"

    with pytest.raises(ValueError) as record:
        write_3_images(file_name, arrays_test[0], arrays_test[1], arrays_test[2], order)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )

    order = "random"

    with pytest.raises(ValueError) as record:
        _ = read_3_images(file_name, order)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )


def test_swaps(arrays_test):

    """
    This series of checks are meant to check whether an array in column major
    order can be converted back to row major order.

    Here arrays containing only zeros or only ones are used because when
    they will be converted to binary format and read in column major or
    row major their values won't change.
    """

    # This is to change the order of two arrays at a time.
    result_a, result_b = column_row_of_two(arrays_test[0], arrays_test[1])

    assert np.allclose(result_a, arrays_test[0])
    assert np.allclose(result_b, arrays_test[1])

    # This is to change the order of three arrays at a time.
    result_a, result_b, result_c = column_row_of_three(arrays_test[0], arrays_test[1], arrays_test[2])

    assert np.allclose(result_a, arrays_test[0])
    assert np.allclose(result_b, arrays_test[1])
    assert np.allclose(result_c, arrays_test[2])
