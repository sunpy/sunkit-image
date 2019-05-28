"""
==================================
Multi-scale Gaussian Normalization
==================================
This example applies Multi-scale Gaussian Normalization to a sunpy map.
The example uses `sunkit_image.enhance` to apply the filter.
"""
# Start by importing the necessary modules.
import matplotlib.pyplot as plt

import sunpy.map
import sunpy.data.sample

import sunkit_image.enhance as enhance

###########################################################################
# Sunpy sample data contains a number of suitable maps, where the sunpy.data.sample.NAME
# returns the location of the given FITS file.

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

###########################################################################
# Applying both the variants of Multi-scale Gaussian Normalization

out = enhance.mgn(aia_map.data)
out = sunpy.map.Map(out, aia_map.meta)

###########################################################################
# The resulting sunpy.map are plotted

fig = plt.figure()
out.plot()
plt.show()
