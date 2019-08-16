"""
===================================
Fourier Local Correlation Tracking
===================================

This example applies Fourier Local Correlation Tracking (FLCT)
to a set of two arrays using `~sunkit_image.flct.flct`.
"""

import numpy as np

from sunkit_image.flct.flct import flct
import matplotlib.pyplot as plt

###########################################################################
# This examples demonstrates how to find the 2D velocity flow field.
# We will create two dummy images containing a moving object and
# assume the time difference between the images to be one second.

# Creating the input arrays
image1 = np.zeros((10, 10))
image1[0:3, 0:3] = 1

image2 = np.zeros((10, 10))
image2[1:4, 1:4] = 1

# Plot both the images
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121)
plt.imshow(image1)
ax1.set_title("First Image")

ax2 = fig.add_subplot(122)
plt.imshow(image2)
ax2.set_title("Second Image")

###########################################################################
# Now we come to the main function where FLCT is applied.
# The values of the parameters were used, were the ones
# which gave the best visual result of the velocities.
vel_x, vel_y, vm = flct(image1, image2, 1, 1, 2.3)

###########################################################################
# The return values are the two dimensional velocity field with ``vel_x``
# showing velocity in the x-direction and ``vel_y``, the velocity in the
# y-direction. ``vm`` is the mask array which shows the pixel
# locations where the FLCT calculations were done. We will also plot the velocity
# in both 'x' and 'y' directions individually.

# First plot the velocity in ``x`` direction
fig = plt.figure()
plt.title("Velocity in x direction")
plt.imshow(vel_x)

###########################################################################
# We can clearly see that all the points have a constant velocity equal to ``1``
# in the ``x`` direction. So if we move our original image ``1`` pixel in the positive
# ``x`` we will get the following plot

# Moving the original image 1 pixel towards right
image1[0:3, 0] = 0
image1[0:3, 1:4] = 1

# Plot the shifted image
fig = plt.figure()
plt.title("Original image shifted by 1 pixel in the X direction")
plt.imshow(image1)

##########################################################################
# Now we will see the effect of velocity in the ``y`` direction.

# First, plot the velocity in ``y`` direction
fig = plt.figure()
plt.title("Velocity in y direction")
plt.imshow(vel_y)

###########################################################################
# We can clearly see that all the points have a constant velocity equal to ``1``
# in the ``y`` direction. So if we move our previously shifted image ``1`` pixel in
# the ``y`` we can account for the velocity in ``y`` direction.

# Moving the original image 1 pixel towards right
image1[0, 1:4] = 0
image1[1:4, 1:4] = 1

# Plot the shifted image
fig = plt.figure()
plt.title("Previously shifted image shifted by 1 pixel in the Y direction")
plt.imshow(image1)

############################################################################
# Since we have accounted for the velocities in both ``x`` and ``y`` direction,
# the shifted `image1` should be same as the final image, `image2`. We can
# visualize that by plotting both of them side by side.

# Plot both the images
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121)
plt.imshow(image1)
ax1.set_title("Shifted First Image")

ax2 = fig.add_subplot(122)
plt.imshow(image2)
ax2.set_title("Second Image")

################################################################################
# You must have noticed that both the velocities in ``x`` and ``y`` have similar
# plots with most of the pixel values having values equal to 1. This denotes that
# each pixel has a velocity of 1 in both the direction. So net velocity of each
# velocity is diagonal which should be the case as evident from the two input images.
# Notice the lonely pixel which has value less than zero. This discrepancy is due
# one of the limitations of the `flct` algorithm.

# Show all the plots
plt.show()
