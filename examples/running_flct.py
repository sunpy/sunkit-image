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
# FLCT functions. This entire example has three parts all of which depicts motion
# of the image in some particular directions.

#############################################################################
# First we plot the velocity field when the image moves in positive ``X`` direction only
 
# Creating the input arrays
image1 = np.zeros((10, 10))
image1[0:3, 0:3] = 1

image2 = np.zeros((10, 10))
image2[0:3, 1:4] = 1

###########################################################################
# Now we come to the main function where FLCT is applied.
# The values of the parameters were used which gave the best visual result
# of the velocities. The time difference between the two images, ``deltat`` is assumed
# to be 1 second. The units of length of the side of a single pixel, ``deltas`` is assumed
# to be 1. The width of Gaussian used to weigh the subimages, ``sigma`` is taken to be 2.3.
# Note you should always experiment with the values of ``sigma`` to get the best results.
vel_x, vel_y, vm = flct.flct(image1, image2, 1, 1, 2.3)

############################################################################
# We will plot the 2D flow field what we get when the image have moved in positive ``X`` direction

# But first we need to create a meshgrid on which the flow field will be plotted
X = np.arange(0, 10, 1)
Y = np.arange(0, 10, 1)
U, V = np.meshgrid(X, Y)

# Plotting the first image
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
plt.imshow(image1, origin="lower")
ax1.set_title("First Image")

# Plot the 2D flow field
ax2 = fig.add_subplot(132)
ax2.quiver(U, V, vel_x, vel_y, scale=20)
ax2.set_title("Flow Field")

# Plot the shifted image
ax3 = fig.add_subplot(133)
plt.imshow(image2, origin="lower")
ax3.set_title("Second Image")

#############################################################################
# Nw, we plot the velocity field when the image moves in positive ``Y`` direction only
 
# Creating the input arrays
image1 = np.zeros((10, 10))
image1[0:3, 0:3] = 1

image2 = np.zeros((10, 10))
image2[1:4, 0:3] = 1

###########################################################################
# Now we come to the main function where FLCT is applied.
# The values of the parameters were used which gave the best visual result
# of the velocities. The time difference between the two images, ``deltat`` is assumed
# to be 1 second. The units of length of the side of a single pixel, ``deltas`` is assumed
# to be 1. The width of Gaussian used to weigh the subimages, ``sigma`` is taken to be 2.3.
# Note you should always experiment with the values of ``sigma`` to get the best results.
vel_x, vel_y, vm = flct.flct(image1, image2, 1, 1, 2.3)

############################################################################
# We will plot the 2D flow field what we get when the image have moved in positive ``Y`` direction

# Plotting the first image
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
plt.imshow(image1, origin="lower")
ax1.set_title("First Image")

# Plot the 2D flow field
ax2 = fig.add_subplot(132)
ax2.quiver(U, V, vel_x, vel_y, scale=20)
ax2.set_title("Flow Field")

# Plot the shifted image
ax3 = fig.add_subplot(133)
plt.imshow(image2, origin="lower")
ax3.set_title("Second Image")

#############################################################################
# Finally we plot the velocity field when the image moves in both the directions
 
# Creating the input arrays
image1 = np.zeros((10, 10))
image1[0:3, 0:3] = 1

image2 = np.zeros((10, 10))
image2[1:4, 1:4] = 1

###########################################################################
# Now we come to the main function where FLCT is applied.
# The values of the parameters were used which gave the best visual result
# of the velocities. The time difference between the two images, ``deltat`` is assumed
# to be 1 second. The units of length of the side of a single pixel, ``deltas`` is assumed
# to be 1. The width of Gaussian used to weigh the subimages, ``sigma`` is taken to be 2.3.
# Note you should always experiment with the values of ``sigma`` to get the best results.
vel_x, vel_y, vm = flct.flct(image1, image2, 1, 1, 2.3)

############################################################################
# We will plot the 2D flow field what we get when the image have moved in both directions

# Plotting the first image
fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
plt.imshow(image1, origin="lower")
ax1.set_title("First Image")

# Plot the 2D flow field
ax2 = fig.add_subplot(132)
ax2.quiver(U, V, vel_x, vel_y, scale=20)
ax2.set_title("Flow Field")

# Plot the shifted image
ax3 = fig.add_subplot(133)
plt.imshow(image2, origin="lower")
ax3.set_title("Second Image")

################################################################################
# Notice the lonely pixel in the bottom left of every flow field which has value less than zero.
# This discrepancy is due one of the limitations of the `flct` algorithm where it is unable to reliably
# calculate the velocitites within ``sigma`` pixels of the image edges.

# Show all the plots
plt.show()
