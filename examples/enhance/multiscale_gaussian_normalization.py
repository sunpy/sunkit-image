"""
==================================
Multi-scale Gaussian Normalization
==================================
This example applies both the variants of Multi-scale
Gaussian Normalization to a sunpy map.
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

out1 = enhance.mgn1(aia_map.data)
out1 = sunpy.map.Map(out1, aia_map.meta)

out2 = enhance.mgn2(aia_map.data)
out2 = sunpy.map.Map(out2, aia_map.meta)

###########################################################################
# The resulting sunpy.map are plotted

fig = plt.figure(figsize=(1,2))

ax1 = fig.add_subplot(121, projection=aia_map)
out1.plot(axes=ax1)

ax2 = fig.add_subplot(122, projection=aia_map)
out2.plot(axes=ax2)
plt.show()
