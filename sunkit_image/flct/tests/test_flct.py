import os
import sys

import numpy as np
import pytest

import sunkit_image.data.test as data
import sunkit_image.flct as flct

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

    file_name = "temp.dat"

    flct.write_2_images(file_name, arrays_test[0], arrays_test[1])

    arr, barr = flct.read_2_images(file_name)

    os.remove(file_name)

    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))

    flct.write_2_images(file_name, arrays_test[0], arrays_test[1], order="column")

    arr, barr = flct.read_2_images(file_name, order="column")

    os.remove(file_name)

    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))

    order = "random"

    with pytest.raises(ValueError) as record:
        flct.write_2_images(file_name, arrays_test[0], arrays_test[1], order)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )

    order = "random"

    with pytest.raises(ValueError) as record:
        _ = flct.read_2_images(file_name, order)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )


def test_three_read_write(arrays_test):

    file_name = "temp.dat"

    flct.write_3_images(file_name, arrays_test[0], arrays_test[1], arrays_test[2])

    arr, barr, carr = flct.read_3_images(file_name)

    os.remove(file_name)

    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))
    assert np.allclose(carr, np.zeros((4, 4)))

    flct.write_3_images(file_name, arrays_test[0], arrays_test[1], arrays_test[2], order="column")

    arr, barr, carr = flct.read_3_images(file_name, order="column")

    os.remove(file_name)

    assert np.allclose(arr, np.zeros((4, 4)))
    assert np.allclose(barr, np.ones((4, 4)))
    assert np.allclose(carr, np.zeros((4, 4)))

    order = "random"

    with pytest.raises(ValueError) as record:
        flct.write_3_images(file_name, arrays_test[0], arrays_test[1], arrays_test[2], order)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )

    order = "random"

    with pytest.raises(ValueError) as record:
        _ = flct.read_3_images(file_name, order)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )


def test_swaps(arrays_test):

    result_a, result_b = flct.column_row_of_two(arrays_test[0], arrays_test[1])

    assert np.allclose(result_a, arrays_test[0])
    assert np.allclose(result_b, arrays_test[1])

    result_a, result_b, result_c = flct.column_row_of_three(
        arrays_test[0], arrays_test[1], arrays_test[2]
    )

    assert np.allclose(result_a, arrays_test[0])
    assert np.allclose(result_b, arrays_test[1])
    assert np.allclose(result_c, arrays_test[2])


# Testing the main FLCT function


@pytest.fixture
def images():

    filepath1 = data.get_test_filepath("hashgauss_F1.csv")
    filepath2 = data.get_test_filepath("hashgauss_F2.csv")

    # These CSV files were created using the IDL IO routines but their order is
    # is not swapped here because we will do it in the flct function.
    image1 = np.genfromtxt(filepath1, delimiter=",")
    image2 = np.genfromtxt(filepath2, delimiter=",")

    return (image1, image2)


@pytest.fixture
def images_dat():

    filepath1 = data.get_test_filepath("hashgauss.dat")
    arr, barr = flct.read_2_images(filepath1)

    # The arrays are directly read from the dat files using the python functions
    # so there is no need to swap their order as they are already in row major.
    return (arr, barr)


@pytest.fixture
def outputs_dat():

    filepath1 = data.get_test_filepath("testgaussvel.dat")
    arr, barr, carr = flct.read_3_images(filepath1)

    # The arrays are directly read from the dat files using the python functions
    # so there is no need to swap their order as they are already in row major.
    return (arr, barr, carr)


@pytest.fixture
def outputs():

    filepath_x = data.get_test_filepath("testgauss_vx.csv")
    filepath_y = data.get_test_filepath("testgauss_vy.csv")
    filepath_m = data.get_test_filepath("testgauss_vm.csv")

    expect_x = np.genfromtxt(filepath_x, delimiter=",")
    expect_y = np.genfromtxt(filepath_y, delimiter=",")
    expect_m = np.genfromtxt(filepath_m, delimiter=",")

    # Since these CSV files were created by reading the dat file on FLCT website using the IDL IO
    # routines their order needs to be rectified.
    expect_x, expect_y, expect_m = flct.column_row_of_three(expect_x, expect_y, expect_m)

    return (expect_x, expect_y, expect_m)


def test_flct_array(images, outputs):

    vx, vy, vm = flct.flct(images[0], images[1], 1, 1, 5, "column", kr=0.5)

    assert np.allclose(vx, outputs[0], atol=1e-5, rtol=1e-6)
    assert np.allclose(vy, outputs[1], atol=1e-5, rtol=1e-6)
    assert np.allclose(vm, outputs[2], atol=1e-5, rtol=1e-6)

    # Errors checks
    order = "random"

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images[0], images[1], 1, 1, 5, order, kr=0.5)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images[0], images[1], 1, 1, 5, kr=0.5, skip=-1)

    assert str(record.value) == "Skip value must be greater than zero."

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images[0], images[1], 1, 1, 5, kr=0.5, skip=1, xoff=4)

    assert str(record.value) == "The absolute value of 'xoff' and 'yoff' must be less than skip."

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images[0], images[1], 1, 1, 5, kr=40)

    assert str(record.value) == "The value of 'kr' must be between 0 and 20."

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images[0], images[1], 1, 1, 5, kr=0.5, skip=1000)

    assert str(record.value) == "Skip is greater than the input dimensions"

    # These tests are dummy tests. They are written just to make sure that FLCT runs on optional
    # parameters also. We did not have any values to compare our results against so this is why
    # these tests are at the end such that only after all the valid tests are passed then these
    # are executed. These are not tests in the strictest sense rather it is designed to increase
    # the test coverage for lines containing the setting of optional arguments.

    _ = flct.flct(
        images[0], images[1], 1, 1, 0, interp=True, quiet=True, absflag=True, biascor=True, pc=True
    )
    # _ = flct.flct(	
    #     images[0],	
    #     images[1],	
    #     1,	
    #     1,	
    #     4,	
    #     skip=4,	
    #     interp=True,	
    #     quiet=True,	
    #     absflag=True,	
    #     biascor=True,	
    #     pc=True,	
    #     xoff=-2,
    #     yoff=-2,
    # )
    assert True


def test_flct_dat(images_dat, outputs_dat):

    vx, vy, vm = flct.flct(images_dat[0], images_dat[1], 1, 1, 5, kr=0.5)

    assert np.allclose(vx, outputs_dat[0])
    assert np.allclose(vy, outputs_dat[1])
    assert np.allclose(vm, outputs_dat[2])

    # Errors checks
    order = "random"

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images_dat[0], images_dat[1], 1, 1, 5, order, kr=0.5)

    assert (
        str(record.value)
        == "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
    )

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images_dat[0], images_dat[1], 1, 1, 5, kr=0.5, skip=-1)

    assert str(record.value) == "Skip value must be greater than zero."

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images_dat[0], images_dat[1], 1, 1, 5, kr=0.5, skip=1, xoff=4)

    assert str(record.value) == "The absolute value of 'xoff' and 'yoff' must be less than skip."

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images_dat[0], images_dat[1], 1, 1, 5, kr=40)

    assert str(record.value) == "The value of 'kr' must be between 0 and 20."

    with pytest.raises(ValueError) as record:
        _ = flct.flct(images_dat[0], images_dat[1], 1, 1, 5, kr=0.5, skip=1000)

    assert str(record.value) == "Skip is greater than the input dimensions"

    # These tests are dummy tests. They are written just to make sure that FLCT runs on optional
    # parameters also. We did not have any values to compare our results against so this is why
    # these tests are at the end such that only after all the valid tests are passed then these
    # are executed. These are not tests in the strictest sense rather it is designed to increase
    # the test coverage for lines containing the setting of optional arguments.

    _ = flct.flct(
        images_dat[0],
        images_dat[1],
        1,
        1,
        0,
        interp=True,
        quiet=True,
        absflag=True,
        biascor=True,
        pc=True,
    )
    # _ = flct.flct(	
    #     images_dat[0],	
    #     images_dat[1],	
    #     1,	
    #     1,	
    #     4,	
    #     skip=4,	
    #     interp=True,	
    #     quiet=True,	
    #     absflag=True,	
    #     biascor=True,	
    #     pc=True,	
    #     xoff=-2,	
    #     yoff=-2,	
    # )
    assert True
