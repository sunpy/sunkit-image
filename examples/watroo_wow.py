"""
==================================
Wavelets Optimized Whitening (WOW)
==================================

This example applies Wavelets Optimized Whitening to a `sunpy.map.Map` using `sunkit_image.enhance.wow`.
"""
# sphinx_gallery_thumbnail_number = 2  # NOQA: ERA001

import matplotlib.pyplot as plt

from astropy import units as u

import sunpy.data.sample
import sunpy.map

import sunkit_image.enhance as enhance

###########################################################################
# `sunpy` provides a range of sample data with  a number of suitable images.

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

# The original image is plotted to showcase the difference.
fig = plt.figure()
ax = plt.subplot(projection=aia_map)
aia_map.plot(clip_interval=(1, 99.99) * u.percent)

###########################################################################
# Applying Wavelets Optimized Whitening on a solar image.
# The `sunkit_image.enhance.wow` function takes either a `sunpy.map.Map` or a `numpy.ndarray` as a input.
# We will use the bilateral flavor of the algorithm, and denoising coefficients in the first three wavelet
# planes equal to 5, 2, & 1 sigma of the local noise. The noise is estimated automatically.
# It is possible to pass a noise map for more optimal results.

wow_map = enhance.wow(aia_map, bilateral=1, denoise_coefficients=[5, 2, 1])

###########################################################################
# Now we will plot the final result.

fig = plt.figure()
ax = plt.subplot(projection=wow_map)
wow_map.plot(clip_interval=(1, 99.99) * u.percent)

plt.show()
