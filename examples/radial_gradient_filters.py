"""
=======================
Radial Gradient Filters
=======================

This example applies both the normalizing radial gradient (`sunkit_image.radial.nrgf`) filter and Fourier
normalizing radial gradient filter (`sunkit_image.radial.fnrgf`) to a sunpy map.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.data.sample
import sunpy.map

import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins

###########################################################################
# `sunpy` sample data contain a number of suitable FITS files for this purpose.
# Here we will use a sample AIA 171 image.

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

###########################################################################
# Both the NRGF and FNRGF work on radial segments above their application radius.
#
# Here we create those segments radial segments. Each segment created will be of
# equal dimensions radially. The distance between 1 solar radii and 2 solar radii
# is divided into 100 equal parts by the following two lines.

radial_bin_edges = equally_spaced_bins()
radial_bin_edges *= u.R_sun

basic_nrgf = radial.nrgf(aia_map, radial_bin_edges)

###########################################################################
# We will need to work out  a few parameters for the FNRGF.
#
# Order is the number of Fourier coefficients to be used in the approximation.
# The attenuation coefficient are calculated to be linearly decreasing, you should
# choose them according to your requirements.

order = 20
attenuation_coefficients = radial.set_attenuation_coefficients(order)

basic_fnrgf = radial.fnrgf(aia_map, radial_bin_edges, order, attenuation_coefficients)

###########################################################################
# Finally we will plot the filtered maps with the original to demonstrate the effect.
#
# The filtered images can be a little washed out so you may need to change some plotting settings
# for a clearer output.

fig = plt.figure(figsize=(7, 15))

ax = fig.add_subplot(311, projection=aia_map)
aia_map.plot(axes=ax, clip_interval=(1, 99.99) * u.percent)

ax1 = fig.add_subplot(312, projection=basic_nrgf, sharey=ax, sharex=ax)
basic_nrgf.plot(axes=ax1, clip_interval=(1, 99.99) * u.percent)
ax1.set_title("NRGF")

ax2 = fig.add_subplot(313, projection=basic_fnrgf, sharey=ax, sharex=ax)
basic_fnrgf.plot(axes=ax2, clip_interval=(1, 99.99) * u.percent)
ax2.set_title("FNRGF")

fig.tight_layout()

plt.show()
