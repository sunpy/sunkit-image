"""
==================================
Multi-scale Gaussian Normalization
==================================

This example applies Multi-scale Gaussian Normalization 
to a SunPy Map using `sunkit_image.enhance.mgn`.
"""
import matplotlib.pyplot as plt

import sunpy.map
import sunpy.data.sample

import sunkit_image.enhance as enhance

###########################################################################
# SunPy sample data contains a number of suitable images, which we will use here.
aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

###########################################################################
# Applying Multi-scale Gaussian Normalization on a solar image.
# The `sunkit_image.enhance.mgn` function takes a `numpy.ndarray` as a input so we will pass only
# the data part of `~sunpy.map.GenericMap`
out = enhance.mgn(aia_map.data)

# The value returned is also a numpy.ndarray so we convert it back to
# a  sunpy.map.GenericMap.
out = sunpy.map.Map(out, aia_map.meta)

###########################################################################
# The resulting map is plotted.
out.plot()
plt.show()
