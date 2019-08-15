"""
===================================
Fourier Local Correlation Tracking
===================================

This example applies Fourier Local Correlation Tracking (FLCT)
to a set of two images taken within a short interval of each other
using `~sunkit_image.flct.flct`.
"""

import numpy as np

import sunkit_image.flct.flct as flct
import matplotlib.pyplot as plt

###########################################################################
# This examples demonstrates how to find the
# 2D velocity flow field between two images taken within a short span of time.
# We will create two dummy images containing a moving
# object and assume the time difference between the images to be one second.

# Creating the input arrays
image1 = np.zeros((10, 10))
image1[0:3, 0:3] = 1

image2 = np.zeros((10, 10))
image2[1:4, 1:4] = 1

# Plot both the images
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121)
plt.imshow(image1)

ax2 = fig.add_subplot(122)
plt.imshow(image2)

###########################################################################
# Now we come to the main function where FLCT is applied. We will pass the
# above arrays to `sunkit_image.flct.flct`

# Since the input arrays were stored in a row major format, no order swapping
# needs to take place. The values of the parameters were used, were the ones
# which gave the best visual result of the velocities.
vel_x, vel_y, vm = flct.flct(image1, image2, 1, 1, 2.3)

###########################################################################
# The return values are the two dimensional velocity field with ``vel_x``
# showing velocity in the x-direction and ``vel_y``, the velocity in the
# y-direction. ``vm`` is the mask array which shows the pixel
# locations where the FLCT calculations were done.

# We will also plot the velocity in both 'x' and 'y' directions
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121)
plt.imshow(vel_x)

ax2 = fig.add_subplot(122)
plt.imshow(vel_y)

################################################################################
# You must have noticed that both the velocities in ``x`` and ``y`` have similar
# plots with most of the pixel values having values equal to 1. This denotes that
# each pixel has a velocity of 1 in both the direction. So net velocity of each
# velocity is diagonal which should be the case as evident from the two input images.
# Notice the lonely pixel which has value less than zero. This discrepancy is due
# one of the limitations of the `flct` algorithm.

# Show all the plots
plt.show()
