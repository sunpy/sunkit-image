"""
================
Fourier Normalizing Radial Gradient Filter
================

This example applies fourier normalizing gradient filter to a sunpy map.

The example uses `sunkit_image.offlimb` to apply the filter.

"""
# Start by importing the necessary modules.
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

import sunpy.map
import sunpy.data.sample

import sunkit_image.offlimb as offlimb
from sunkit_image.utils import equally_spaced_bins

###########################################################################
# Sunpy sample data contains a number of suitable maps, where the sunpy.data.sample.NAME
# returns the location of the given FITS file.

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

###########################################################################
# Radial bins are calculated over which the FNRGF filter will be applied.
# The FNRGF filter is applied, after assuming values for the parameters.

radial_bin_edges = equally_spaced_bins()
radial_bin_edges *= u.R_sun

order = 20
attenuation_coefficients = np.zeros((2, order + 1))
attenuation_coefficients[0,:] = np.linspace(1,0,order + 1)
attenuation_coefficients[1,:] = np.linspace(1,0,order + 1)

out = offlimb.fnrgf(aia_map, radial_bin_edges, order, attenuation_coefficients)

###########################################################################
# The resulting sunpy.map is plotted

fig = plt.figure()
out.plot()
plt.show()
