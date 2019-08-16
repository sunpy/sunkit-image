"""
===================================
Fourier Local Correlation Tracking
===================================

This example applies Fourier Local Correlation Tracking (FLCT)
to a set of two arrays using `~sunkit_image.flct.flct`.
"""

import numpy as np

import matplotlib.pyplot as plt
import sunkit_image.flct as flct

###########################################################################
# This examples demonstrates how to find the 2D velocity flow field.
# We will create two dummy images containing a moving object and
# assume the time difference between the images to be one second.
# Here we will create two dummy images assuming time between them as
# 1 second. We also plot both the images as subplots and demonstrate how
# FLCT functions

# Creating the input arrays
image1 = np.zeros((10, 10))
image1[0:3, 0:3] = 1

image2 = np.zeros((10, 10))
image2[1:4, 1:4] = 1

# Plot both the generated input images
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121)
plt.imshow(image1, origin="lower")
ax1.set_title("First Image")

ax2 = fig.add_subplot(122)
plt.imshow(image2, origin="lower")
ax2.set_title("Second Image")

###########################################################################
# Now we come to the main function where FLCT is applied.
# The values of the parameters were used which gave the best visual result
# of the velocities.
# The set of parameters used:
# * The time difference between the two images, ``deltat`` is assumed to be 1 second.
# * The units of length of the side of a single pixel, ``deltas`` is assumed to be 1.
# * The width of Gaussian used to weigh the subimages, ``sigma`` is taken to be 2.3.
# Note you should always experiment with the values of ``sigma`` to get the best results.
vel_x, vel_y, vm = flct.flct(image1, image2, 1, 1, 2.3)

###########################################################################
# The return values are the two dimensional velocity field with ``vel_x``
# showing velocity in the x-direction and ``vel_y``, the velocity in the
# y-direction. ``vm`` is the mask array which shows the pixel
# locations where the FLCT calculations were done. We will also plot the velocity
# in both 'x' and 'y' directions individually.

# First plot the velocity in ``x`` direction
fig = plt.figure()
plt.title("Velocity in x direction")
plt.imshow(vel_x, origin="lower")

###########################################################################
# We can clearly see that all the points have a constant velocity equal to ``1``
# in the ``x`` direction. Here we will also plot the flow field in the
# ``x`` direction along with the original image and the shifted image if that flow
# field is applied to it.

# Plotting the original image again
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
plt.imshow(image1, origin="lower")
ax1.set_title("Original Image")

# Plot the flow field only in the ``x`` direction assuming the flow field is
# zero in ``y`` direction

# Creating the points
X = np.arange(0, 10, 1)
Y = np.arange(0, 10, 1)
U,V = np.meshgrid(X,Y)

# Plotting the flow field with velocity in ``y`` as zero
ax2 = fig.add_subplot(132)
ax2.quiver(U, V, vel_x, 0, scale=20)
ax2.set_title("Flow Field in the X direction")

# Moving the original image 1 pixel in the positive X direction
image1[0:3, 0] = 0
image1[0:3, 1:4] = 1

# Plot the shifted image
ax3 = fig.add_subplot(133)
plt.imshow(image1, origin="lower")
ax3.set_title("The original image shifted in positve X")

##########################################################################
# Now we will see the effect of velocity in the ``y`` direction.

# First, plot the velocity in ``y`` direction
fig = plt.figure()
plt.title("Velocity in y direction")
plt.imshow(vel_y, origin="lower")

###########################################################################
# We can clearly see that all the points have a constant velocity equal to ``1``
# in the ``y`` direction. Here we will also plot the flow field in the
# ``x`` direction along with the original image and the shifted image if that flow
# field is applied to it.

# Plotting the original image again
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
plt.imshow(image1, origin="lower")
ax1.set_title("Original Image")

# Plot the flow field only in the ``y`` direction assuming the flow field is
# zero in ``x`` direction
ax2 = fig.add_subplot(132)
ax2.quiver(U, V, 0, vel_y, scale=20)
ax2.set_title("Flow Field in the Y direction")

# Moving the original image 1 pixel in the positive Y direction
image1[0, 1:4] = 0
image1[1:4, 1:4] = 1

# Plot the shifted image
ax3 = fig.add_subplot(133)
plt.imshow(image1, origin="lower")
ax3.set_title("The previous image now shifted in positve Y")

############################################################################
# Above we have shown the effect of the flow field when it is present in either ``x`` or ``y`` directions,
# and also demonstrated how they can be used independently to get to the final image.
# But now we will show the effect of the entire flow field acting in both the directions simultaneously. 

# Plotting the original image again
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
image1 = np.zeros((10, 10))
image1[0:3, 0:3] = 1
plt.imshow(image1, origin="lower")
ax1.set_title("Original Image")

# Plotting the two dimensional flow field
ax2 = fig.add_subplot(132)
ax2.quiver(U, V, vel_x, vel_y, scale=20)
ax2.set_title("Flow Field in the both direction")

# Plot the shifted image
ax3 = fig.add_subplot(133)
image2 = np.zeros((10, 10))
image2[1:4, 1:4] = 1
plt.imshow(image2, origin="lower")
ax3.set_title("The original image shifted in velocity direction")

################################################################################
# You must have noticed that both the velocities in ``x`` and ``y`` have similar
# plots with most of the pixel values having values equal to 1. This denotes that
# each pixel has a velocity of 1 in both the direction. So net velocity of each
# velocity is diagonal which should be the case as evident from the two input images.
# Notice the lonely pixel which has value less than zero. This discrepancy is due
# one of the limitations of the `flct` algorithm where it is unable to reliable calculate
# the velocitites within ``sigma`` pixels of the image edges.

# Show all the plots
plt.show()
