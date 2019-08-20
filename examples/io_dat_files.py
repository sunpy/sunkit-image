"""
==================================
Input/Output of dat files for FLCT
==================================

This example demostrates the use of functions in `sunkit_image.flct`
to read and write arrays to binary ``dat`` files.
"""

import numpy as np

import sunkit_image.flct as flct

###########################################################################
# We will create three arrays which we will save out to a ``dat`` file.
# The original FLCT C code uses ``dat`` files for both input and
# output. So these functions can be used to read pre-existing ``dat`` files
# and writing new files which keeps your current work compatible with the
# existing implementation.

a = np.zeros((4, 4))
b = np.ones((4, 4))
c = np.arange(16).reshape((4, 4))

###########################################################################
# First, we will demonstrate writing to a ``dat`` file.

# We can write two arrays to dat file using flct.write_2_images
flct.write_2_images("two.dat", a, b)

# Three arrays can also be written to a dat file using flct.write_3_images
flct.write_3_images("three.dat", a, b, c)

###########################################################################
# We can get back these arrays by using the read functions in `sunkit_image.flct`
# It is to be noted that these read functions can only read ``dat``
# files, the ones which were written using `~sunkit_image.flct.write_2_images`,
# `~sunkit_image.flct.read_3_images` and the IDL IO routines as given on the
# FLCT `website <http://cgem.ssl.berkeley.edu/cgi-bin/cgem/FLCT/dir?ci=tip>`__.

# Reading two arrays from a dat file
one, two = flct.read_2_images("two.dat")

# Reading three arrays from a dat file
one, two, three = flct.read_3_images("three.dat")

###########################################################################
# We can verify that the arrays written and the arrays we read in are the same.

assert np.allclose(one, a)
assert np.allclose(two, b)
assert np.allclose(three, c)
