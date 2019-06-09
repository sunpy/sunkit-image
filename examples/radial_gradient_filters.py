"""
=======================
Radial Gradient Filters
=======================

This example applies both the normalizing radial gradient (`sunkit_image.radial.nrgf`) filter and Fourier
normalizing radial gradient filter (`sunkit_image.radial.fnrgf`) to a sunpy map.


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

# Radial bins are calculated over which the filters will be applied. The distance
# between 1 Solar radius and 2 Solar radius is divided into 100 equal parts.
radial_bin_edges = equally_spaced_bins()
radial_bin_edges *= u.R_sun

# The NRGF filter is applied after it.
out1 = radial.nrgf(aia_map, radial_bin_edges)

# The NRGF filtered map is plotted. 
out1.plot()
plt.show()

###########################################################################

# Assuming values for the parameters of FNRGF
# Order is the number of Fourier coefficients to be used in the approximation.
# The attentuation coefficient are calculated to be linearly decreasing, you should
# choose them according to your requirements.
order = 20
attenuation_coefficients = np.zeros((2, order + 1))
attenuation_coefficients[0, :] = np.linspace(1, 0, order + 1)
attenuation_coefficients[1, :] = np.linspace(1, 0, order + 1)

# The FNRGF filter is applied after it.
out2 = radial.fnrgf(aia_map, radial_bin_edges, order, attenuation_coefficients)

# The FNRGF filtered map is plotted. 
out2.plot()
plt.show()
