"""
===================================================
Coaligning AIA and EIS data with cross-correlations
===================================================

This example shows how to coalign AIA and EIS data using the method of cross-correlations which is implemented in the match_template function within the sunkit-image.
Here we have an EIS raster image which has an incorrect pointing. This would be corrected using the AIA image as a reference. 

"""

import matplotlib.pyplot as plt

import astropy.units as u
from sunpy.net import Fido ,attrs as a
import sunpy.map
from sunkit_image.coalignment import coalignment
from sunkit_image.data.test import get_test_filepath

###################################################################################
# Firstly, let's load up the EIS raster and see what it looks like.
# This raster in particular doesn't have the correct pointing information.

eis_map = sunpy.map.Map(get_test_filepath("eis_20140108_095727.fe_12_195_119.2c-0.int.fits"))
eis_map.plot()

# Lets find the AIA image that we want to use as a reference. We would be using 
# the image near the date_average of this raster.

date_avg = eis_map.meta["date_avg"]
date_start = eis_map.meta["date_beg"]
date_end = eis_map.meta["date_end"]

aia_193_full_disc_map = sunpy.map.Map(Fido.fetch(Fido.search(a.Time(start = date_start,near=date_avg, end=date_end), a.Instrument('aia'), a.Wavelength(193*u.angstrom))))

####################################################################################
# Before coaligning the images, we first downsample the AIA image to the same plate 
# scale as the EIS image.

nx= (aia_193_full_disc_map.scale.axis1 * aia_193_full_disc_map.dimensions.x )/eis_map.scale.axis1
ny= (aia_193_full_disc_map.scale.axis2 * aia_193_full_disc_map.dimensions.y )/eis_map.scale.axis2

aia_193_downsampled_map = aia_193_full_disc_map.resample(u.Quantity([nx,ny]))

####################################################################################
# Now we can coalign the EIS and AIA images using the cross-correlation method. For 
# this we would be using the match_template function within the sunkit_image.coalignment 
# module. For more details for the implementation of this function, please refer to the
# documentation of `~sunkit_image.coalignment.match_template`.

coaligned_eis_map = coalignment(aia_193_downsampled_map, eis_map, "match_template")

####################################################################################
# Now for better visualization of the coaligned images, we would be plotting the EIS 
# contours of the bright regions over the AIA image before and after coalignment. 

levels = [200, 400, 500, 700, 800] * aia_193_full_disc_map.unit
fig = plt.figure(figsize=(15,7.5))
ax = fig.add_subplot(121, projection=eis_map)
eis_map.plot(axes=ax, title='Before pointing correction')
bounds = ax.axis()
aia_193_full_disc_map.draw_contours(levels, axes=ax, cmap='sdoaia171', alpha=0.3)
ax.axis(bounds)
ax = fig.add_subplot(122, projection=coaligned_eis_map)
coaligned_eis_map.plot(axes=ax, title='After pointing correction')
bounds = ax.axis()
aia_193_full_disc_map.draw_contours(levels, axes=ax, cmap='sdoaia171', alpha=0.3)
ax.axis(bounds)