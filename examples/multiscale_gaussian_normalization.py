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

# Applying both the variants of Multi-scale Gaussian Normalization
# The ``mgn`` function takes a ``numpy.ndarray`` as a input so we will pass only
# the data part of ``sunpy.map.Map``
out = enhance.mgn(aia_map.data)

# The value returned is also a ``numpy.ndarray`` so we convert it back to
# ``sunpy.map.Map``
out = sunpy.map.Map(out, aia_map.meta)

###########################################################################

# The resulting Map is plotted.
out.plot()
plt.show()
