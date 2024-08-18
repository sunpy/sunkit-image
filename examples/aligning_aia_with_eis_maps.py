"""
=====================
Coaligning EIS to AIA
=====================

This example shows how to EISA data to AIA using cross-correlation which is implemented as the "match_template" method.
"""

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from sunpy.net import Fido ,attrs as a
import sunpy.map
from sunkit_image.coalignment import coalign
from sunkit_image.data.test import get_test_filepath

###################################################################################
# Firstly, let us acquire the EIS and AIA data we need for this example.

eis_map = sunpy.map.Map(get_test_filepath("eis_20140108_095727.fe_12_195_119.2c-0.int.fits"))
eis_map.plot()

###################################################################################
# Lets find the AIA image that we want to use as a reference.
# We would be using the image near the date_average of this raster.

# The following way is necessary because this eis doesn't have direct date_start, date_avg and date_end attributes.
query = Fido.search(a.Time(start=eis_map.meta["date_beg"], near=eis_map.meta["date_avg"], end=eis_map.meta["date_end"]), a.Instrument('aia'), a.Wavelength(193*u.angstrom))
aia_file = Fido.fetch(query)
aia_map = sunpy.map.Map(aia_file)

####################################################################################
# Before coaligning the images, we first downsample the AIA image to the same plate 
# scale as the EIS image. This is not done automatically.

nx = (aia_map.scale.axis1 * aia_map.dimensions.x) / eis_map.scale.axis1
ny = (aia_map.scale.axis2 * aia_map.dimensions.y) / eis_map.scale.axis2

aia_downsampled = aia_map.resample(u.Quantity([nx, ny]))

####################################################################################
# Now we can coalign EIS to AIA using a cross-correlation.
# For this we would be using the "match_template" method.
# For details of the implementation refer to the
# documentation of `~sunkit_image.coalignment.match_template.match_template_coalign`.

coaligned_eis_map = coalign(aia_downsampled, eis_map)

####################################################################################
# To check now effective this has been, we will plot the EIS data and
# overlap the bright regions from AIA before and after coalignment. 
# Plot the EIS data and overlap the bright regions from AIA before and after coalignment.

levels = [200, 400, 500, 700, 800] * aia_map.unit

fig = plt.figure(figsize=(15, 7.5))

# Plot before coalignment
ax = fig.add_subplot(121, projection=eis_map)
eis_map.plot(axes=ax, title='Before', aspect=eis_map.meta['cdelt2'] / eis_map.meta['cdelt1'],
             cmap='Blues_r', norm=ImageNormalize(stretch=AsinhStretch()))
aia_map.draw_contours(levels, axes=ax, alpha=0.3)

# Plot after coalignment
ax = fig.add_subplot(122, projection=coaligned_eis_map)
coaligned_eis_map.plot(axes=ax, title='After', aspect=coaligned_eis_map.meta['cdelt2'] / coaligned_eis_map.meta['cdelt1'],
             cmap='Blues_r', norm=ImageNormalize(stretch=AsinhStretch()))
aia_map.draw_contours(levels, axes=ax, alpha=0.3)

fig.tight_layout()
plt.show()