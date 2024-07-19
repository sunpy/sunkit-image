"""
==================================
Multi-scale Gaussian Normalization
==================================

This example applies Multi-scale Gaussian Normalization to a `sunpy.map.Map` using `sunkit_image.enhance.mgn`.
"""
# sphinx_gallery_thumbnail_number = 2  # NOQA: ERA001

import matplotlib.pyplot as plt
from matplotlib import colors

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
# Applying Multi-scale Gaussian Normalization on a solar image.
# The `sunkit_image.enhance.mgn` function takes either a `sunpy.map.Map` or a `numpy.ndarray` as a input.

mgn_map = enhance.mgn(aia_map)

###########################################################################
# Now we will plot the MGN enhanced map.

fig = plt.figure()
ax = plt.subplot(projection=mgn_map)
mgn_map.plot(norm=colors.Normalize())

plt.show()
