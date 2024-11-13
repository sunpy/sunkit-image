"""
=====================
Coaligning EIS to AIA
=====================

This example shows how to EIS data to AIA using cross-correlation which is implemented as the "match_template" method.
"""
# sphinx_gallery_thumbnail_number = 2 # NOQA: ERA001

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.visualization import AsinhStretch, ImageNormalize

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

from sunkit_image.coalignment import coalign

###################################################################################
# Firstly, let us acquire the EIS and AIA data we need for this example.
#
# For this example, we will use the EIS data from the sunpy data repository.
# This is a preprocessed EIS raster data.


eis_map = sunpy.map.Map("https://github.com/sunpy/data/raw/main/sunkit-image/eis_20140108_095727.fe_12_195_119.2c-0.int.fits")

fig = plt.figure()

ax = fig.add_subplot(111, projection=eis_map)
eis_map.plot(axes=ax, aspect=eis_map.meta['cdelt2'] / eis_map.meta['cdelt1'],
             cmap='Blues_r', norm=ImageNormalize(stretch=AsinhStretch()))

###################################################################################
# Lets find the AIA image that we want to use as a reference.
# We want to be using an image near the "date_average" of the EIS raster.

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
# Now we can coalign EIS to AIA using cross-correlation. For this we would be using the
# "match_template" method. For details of the implementation refer to the
# documentation of `~sunkit_image.coalignment.match_template.match_template_coalign`.

coaligned_eis_map = coalign(aia_downsampled, eis_map)

####################################################################################
# To check now effective this has been, we will plot the EIS data and
# overlap the bright regions from AIA before and after the coalignment.

levels = [800] * aia_map.unit

fig = plt.figure(figsize=(15, 7.5))

# Before coalignment
ax = fig.add_subplot(121, projection=eis_map)
eis_map.plot(axes=ax, title='Before coalignment',
             aspect=coaligned_eis_map.meta['cdelt2'] / coaligned_eis_map.meta['cdelt1'],
             cmap='Blues_r', norm=ImageNormalize(stretch=AsinhStretch()))
aia_map.draw_contours(levels, axes=ax)

# After coalignment
ax = fig.add_subplot(122, projection=coaligned_eis_map)
coaligned_eis_map.plot(axes=ax, title='After coalignment',
                       aspect=coaligned_eis_map.meta['cdelt2'] / coaligned_eis_map.meta['cdelt1'],
                       cmap='Blues_r', norm=ImageNormalize(stretch=AsinhStretch()))
aia_map.draw_contours(levels, axes=ax)

fig.tight_layout()

plt.show()
