"""
===================================
Fourier Linear Correlation Tracking
===================================

This example applies Fourier Linear Correlation Tracking (FLCT)
to a set of two images taken within a short interval of each other
using `sunkit_image.flct.flct`.
"""

import numpy as np

from sunkit_image.flct import flct
import sunkit_image.data.test as data
import sunkit_image.flct._pyflct as pyflct

###########################################################################
# First we need to get the input images which will be read by the
# `pyflct.read_two_images` from a `dat` file stored in `sunkit_image.data.test`

filepath = data.get_test_filepath("hashgauss.dat")
flag, image1, image2 = pyflct.read_two_images(filepath)

###########################################################################
# Now we come to the main function where FLCT is applied. We will pass the
# above read arrays to `sunkit_image.flct.flct`

# Since the input arrays were stored in a row major format, `row` was passed
# as the `order`
# TODO More explanation on the order
vel_x, vel_y, vel_m = flct(image1, image2, "row", 1, 1, 5, kr=0.5)

###########################################################################
# The return values are the two dimensional velocity field with `vel_x`
# showing velocity in the x-direction and `vel_y`, the velocity in the
# y-direction.
