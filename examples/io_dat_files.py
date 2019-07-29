"""
==================================
Input/Output of dat files for FLCT
==================================

This example demostrates the use of functions in `sunkit_image.flct._pyflct`
to read and write arrays to binary ``dat`` files.
"""

import numpy as np

import sunkit_image.flct._pyflct as pyflct

###########################################################################
# We will create three arrays which we will save out to a ``dat`` file. 

a = np.zeros((4, 4))
b = np.ones((4, 4))
c = np.arange(16).reshape((4, 4))

###########################################################################
# First, we will demonstrate writing to a ``dat`` file.

# We can write two arrays to `dat` file using `pyflct.write_two_images`
pyflct.write_two_images("two.dat", a, b)

# Three arrays can also be written to a `dat` file using `pyflct.write_three_images`
pyflct.write_three_images("three.dat", a, b, c)

###########################################################################
# We can get back these arrays by using the read functions in `pyflct`
# It is to be noted that these read functions can only read ``dat``
# files, the ones which were written using `pyflct.write_two_images`,
# `pyflct.read_three_images` and the IDL IO routines as given on the

# Reading two arrays from a dat file
flag, one, two = pyflct.read_two_images("two.dat")

# Reading three arrays from a dat file
flag, one, two, three = pyflct.read_three_images("three.dat")

###########################################################################
# We can verify that the arrays written and the arrays we read in are the same.

assert np.allclose(one, a)
assert np.allclose(two, b)
assert np.allclose(three, c)
