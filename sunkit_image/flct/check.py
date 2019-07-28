import sunkit_image.flct._pyflct as pyflct
import numpy as np

import sunkit_image.data.test as data
from sunkit_image.flct import flct

# Test 1 : The recommended method to read the data
# C functions wrapped in python are used for IO.
filepath1 = data.get_test_filepath("hashgauss.dat")
ier, nx, ny, arr, barr = pyflct.read_two_images(filepath1)

vx, vy, vm = flct(arr, barr, "row", 1, 1, 5, kr=0.5)

filepath2 = data.get_test_filepath("testgaussvel.dat")
ier, nx, ny, vx_out, vy_out, vm_out = pyflct.read_three_images(filepath2)

assert np.allclose(vx, vx_out)
assert np.allclose(vy, vy_out)
assert np.allclose(vm, vm_out)

# Test 2: If you already have arrays read from IDL
# This method can create some data losses.
filepath1 = data.get_test_filepath("hashgauss_F1.csv")
filepath2 = data.get_test_filepath("hashgauss_F2.csv")

# These CSV files were created using the IDL IO routines but their order is
# is not swapped here because we will do it in the flct function.
image1 = np.genfromtxt(filepath1, delimiter=",")
image2 = np.genfromtxt(filepath2, delimiter=",")

images = (image1, image2)

filepath_x = data.get_test_filepath("testgauss_vx.csv")
filepath_y = data.get_test_filepath("testgauss_vy.csv")
filepath_m = data.get_test_filepath("testgauss_vm.csv")

expect_x = np.genfromtxt(filepath_x, delimiter=",")
expect_y = np.genfromtxt(filepath_y, delimiter=",")
expect_m = np.genfromtxt(filepath_m, delimiter=",")

# Since these CSV files were created by reading the dat file using IDL
# their order needs to be rectified.
expect_x, expect_y, expect_m = pyflct.swap_order_three(expect_x, expect_y, expect_m)

outputs = (expect_x, expect_y, expect_m)

vx, vy, vm = flct(images[0], images[1], "column", 1, 1, 5, kr=0.5)

assert np.allclose(vx, outputs[0], atol=1e-5, rtol=1e-6)
assert np.allclose(vy, outputs[1], atol=1e-5, rtol=1e-6)
assert np.allclose(vm, outputs[2], atol=1e-5, rtol=1e-6)
