"""
==================================
Input/Output of dat files for FLCT
==================================

This example demostrates the use of functions in `sunkit_image.flct._pyflct`
to read and write arrays to binary `dat` files.
"""

import numpy as np

# TODO: This to be imported directly
import sunkit_image.flct._pyflct as pyflct

###########################################################################
# We will create three dummy arrays which will help us demonstrate read
# and write to a `dat` file.
# TODO: Add more explanation to the dat files.

a = np.zeros((4, 4))
b = np.ones((4, 4))
c = np.arange(16).reshape((4, 4))

###########################################################################
# First, we will demonstrate writing to a `dat` file.

# We can write two arrays to `dat` file using `pyflct.write_two_images`
pyflct.write_two_images("two.dat", a, b)

# Three arrays can also be written to a `dat` file using `pyflct.write_three_images`
pyflct.write_three_images("three.dat", a, b, c)

###########################################################################
# We can get back these arrays by using the read functions in `pyflct`
# It is to be noted that these read functions can only read special `dat`
# files, the ones which were written using `pyflct.write_two_images`,
# `pyflct.read_three_images` and the IDL IO routines as given on the
# FLCT website.
# TODO: Add a link to the IDL routines

# TODO: Get rid of the flags
# Reading two arrays from a `dat` file
flag, one, two = pyflct.read_two_images("two.dat")

# Reading three arrays
flag, one, two, three = pyflct.read_three_images("three.dat")

###########################################################################
# Just to verify what was written and what is read you can also run the below
# statements.

assert np.allclose(one, a)
assert np.allclose(two, b)
assert np.allclose(three, c)
