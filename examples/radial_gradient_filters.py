"""
=======================
Radial Gradient Filters
=======================

This example applies both the normalizing radial gradient filter and Fourier
normalizing radial gradient filter to a sunpy map.


"""
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

import sunpy.map
import sunpy.data.sample

import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins

###########################################################################
# Sunpy's sample data contain a number of suitable FITS files for this purpose.

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

###########################################################################
# Radial bins are calculated over which the filters will be applied.

radial_bin_edges = equally_spaced_bins()
radial_bin_edges *= u.R_sun

# The NRGF filter is applied after it.
out1 = radial.nrgf(aia_map, radial_bin_edges)

# Assuming values for parameters of FNRGF
order = 20
attenuation_coefficients = np.zeros((2, order + 1))
attenuation_coefficients[0, :] = np.linspace(1, 0, order + 1)
attenuation_coefficients[1, :] = np.linspace(1, 0, order + 1)

# The FNRGF filter is applied after it.
out2 = radial.fnrgf(aia_map, radial_bin_edges, order, attenuation_coefficients)

###########################################################################
# The resulting SunPy Maps are plotted here.

fig = plt.figure(figsize=(1,2))

ax1 = fig.add_subplot(121, projection=aia_map)
out1.plot(axes=ax1)

ax2 = fig.add_subplot(122, projection=aia_map)
out2.plot(axes=ax2)
plt.show()
