import numpy as np

import pytest

from sunkit_image.flct.flct import flct, vcimagein, vcimageout
import sunkit_image.data as data

filepath1 = data.get_test_filepath("hashgauss_1.csv")
filepath2 = data.get_test_filepath("hashgauss_2.csv")

image1 = np.genfromtxt(filepath1, delimiter=',')
image2 = np.genfromtxt(filepath2, delimiter=',')

filepath_x = data.get_test_filepath("testgauss_vx.csv")
filepath_y = data.get_test_filepath("testgauss_vy.csv")
filepath_m = data.get_test_filepath("testgauss_vm.csv")

expect_x = np.genfromtxt(filepath_x, delimiter=',')
expect_y = np.genfromtxt(filepath_y, delimiter=',')
expect_m = np.genfromtxt(filepath_m, delimiter=',')

images = (image1, image2)
outputs = (expect_x, expect_y, expect_m)

vx, vy, vm = flct(images[0], images[1], 1, 1, 5, kr=0.5)

print(vx)

assert np.allclose(vx, outputs[0])
assert np.allclose(vy, outputs[1])
assert np.allclose(vm, outputs[2])

# filepath_x = data.get_test_filepath("vxtest.csv")
# filepath_y = data.get_test_filepath("vytest.csv")
# filepath_m = data.get_test_filepath("vmtest.csv")

# expect_x = np.genfromtxt(filepath_x, delimiter=',')
# expect_y = np.genfromtxt(filepath_y, delimiter=',')
# expect_m = np.genfromtxt(filepath_m, delimiter=',')

# ex = (expect_x, expect_y, expect_m)
