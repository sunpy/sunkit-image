"""
===================================
Fourier Local Correlation Tracking
===================================

This example applies Fourier Local Correlation Tracking (FLCT)
to a set of two images taken within a short interval of each other
using `sunkit_image.flct.flct <http://solarmuri.ssl.berkeley.edu/~fisher/public/software/FLCT/C_VERSIONS/>`__.
"""

import numpy as np

import sunkit_image.flct as flct
import sunkit_image.data.test as data

###########################################################################
# First we need to get the input images which will be read by the
# `flct.read_two_images` from a `dat` file stored in `sunkit_image.data.test`
# Here in this example we are using a pre-existing dat file directly from
# the FLCT website. This is to show that this module is capable of working
# with pre-existing files also.

filepath = data.get_test_filepath("hashgauss.dat")
image1, image2 = flct.read_2_images(filepath)

###########################################################################
# Now we come to the main function where FLCT is applied. We will pass the
# above read arrays to `sunkit_image.flct.flct`

# Since the input arrays were stored in a row major format, so no swapping
# needs to take place.
vel_x, vel_y, vm = flct.flct(image1, image2, 1, 1, 5, kr=0.5)

###########################################################################
# The return values are the two dimensional velocity field with `vel_x`
# showing velocity in the x-direction and `vel_y`, the velocity in the
# y-direction. The last, ``vm`` is the mask array which shows the pixel
# locations where the FLCT calculations were done.
