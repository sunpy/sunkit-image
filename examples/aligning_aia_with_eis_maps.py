"""
=====================
Coaligning EIS to AIA
=====================

This example shows how to EISA data to AIA using cross-correlation which is implemented as the "match_template" method.
"""

import matplotlib.pyplot as plt

import astropy.units as u
from sunpy.net import Fido ,attrs as a
import sunpy.map
from sunkit_image.coalignment import coalignment
from sunkit_image.data.test import get_test_filepath

###################################################################################
# Firstly, us acquire the EIS and AIA data we need for this example.

eis_map = sunpy.map.Map(get_test_filepath("eis_20140108_095727.fe_12_195_119.2c-0.int.fits"))
eis_map.plot()

# Lets find the AIA image that we want to use as a reference.
# We would be using the image near the date_average of this raster.
date_avg = eis_map.meta["date_avg"]
date_start = eis_map.meta["date_beg"]
date_end = eis_map.meta["date_end"]

query = Fido.search(a.Time(start=date_start, near=date_avg, end=date_end), a.Instrument('aia'), a.Wavelength(193*u.angstrom))
aia_file = Fido.fetch(query)
aia_map = sunpy.map.Map(aia_file)

####################################################################################
# Before coaligning the images, we first downsample the AIA image to the same plate 
# scale as the EIS image. This is not done automatically.

nx = (aia_map.scale.axis1 * aia_map.dimensions.x) / eis_map.scale.axis1
ny = (aia_map.scale.axis2 * aia_map.dimensions.y) / eis_map.scale.axis2

aia_downsampled = aia_map.resample(u.Quantity([nx, ny]))

####################################################################################
# Now we can coalign EIS to AIAusing a cross-correlation.
# For this we would be using the "match_template" method.
# For details of the implementation refer to the
# documentation of `~sunkit_image.coalignment.match_template`.

coaligned_eis_map = coalignment(aia_downsampled, eis_map, "match_template")

####################################################################################
# To check now effective this has been, we will plot the EIS data and
# overlap the bright regions from AIA before and after coalignment. 

levels = [200, 400, 500, 700, 800] * aia_map.unit

fig = plt.figure(figsize=(15, 7.5))
ax = fig.add_subplot(121, projection=eis_map)

eis_map.plot(axes=ax, title='Before')
aia_map.draw_contours(levels, axes=ax, alpha=0.3)

ax = fig.add_subplot(122, projection=coaligned_eis_map)
coaligned_eis_map.plot(axes=ax, title='After')
aia_map.draw_contours(levels, axes=ax, alpha=0.3)

fig.tight_layout()

plt.show()
