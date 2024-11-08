"""
===================================
Normalizing Radial Gradient Filters
===================================

This example showcases the filters found in the "radial" module.

These are:

- Normalizing Radial Gradient Filter (NRGF) (`sunkit_image.radial.nrgf`)
- Fourier Normalizing Radial Gradient Filter (FNRGF) (`sunkit_image.radial.fnrgf`)
- Radial Histogram Equalizing Filter (RHEF) (`sunkit_image.radial.rhef`)
"""

import matplotlib.pyplot as plt

import astropy.units as u

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
# is divided into equal parts by the following two lines.

radial_bin_edges = equally_spaced_bins(1, 2, aia_map.data.shape[0] // 4)
radial_bin_edges *= u.R_sun

base_nrgf = radial.nrgf(aia_map, radial_bin_edges=radial_bin_edges, application_radius=1 * u.R_sun)

###########################################################################
# We will need to work out a few parameters for the FNRGF.
#
# Order is the number of Fourier coefficients to be used in the approximation.
# The attenuation coefficients are calculated to be linearly decreasing, you should
# choose them according to your requirements. These can be changed by tweaking the following keywords: ``mean_attenuation_range`` and ``std_attenuation_range``.

order = 20
base_fnrgf = radial.fnrgf(
    aia_map,
    radial_bin_edges=radial_bin_edges,
    order=order,
    application_radius=1 * u.R_sun
)

###########################################################################
# Now we will also use the final filter, RHEF.

base_rhef = radial.rhef(aia_map, radial_bin_edges=radial_bin_edges, application_radius=1 * u.R_sun)

###########################################################################
# Finally we will plot the filtered maps with the original to demonstrate the effect of each.

fig = plt.figure(figsize=(15, 15))

ax = fig.add_subplot(221, projection=aia_map)
aia_map.plot(axes=ax, clip_interval=(1, 99.99) * u.percent)

ax1 = fig.add_subplot(222, projection=base_nrgf)
base_nrgf.plot(axes=ax1)
ax1.set_title("NRGF")

ax2 = fig.add_subplot(223, projection=base_fnrgf)
base_fnrgf.plot(axes=ax2, clip_interval=(1, 99.99) * u.percent)
ax2.set_title("FNRGF")

ax3 = fig.add_subplot(224, projection=base_rhef)
base_rhef.plot(axes=ax3)
ax3.set_title("RHEF")

ax.coords[0].set_ticklabel_visible(False)
ax1.coords[0].set_ticklabel_visible(False)
ax1.coords[1].set_ticklabel_visible(False)
ax3.coords[1].set_ticklabel_visible(False)

fig.tight_layout()

plt.show()
