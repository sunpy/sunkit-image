"""
=================================
Plotting Sunspots using STARA
=================================

This example demonstrates the use of Sunspot Tracking And Recognition
Algorithm (STARA) in detecting and plotting sunspots.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.io._fits
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

from sunkit_image.stara import stara

###############################################################################
# Firstly, let's download HMI continuum data from the JSOC. Specify the time range of interest.

tstart = "2023-01-01 00:00"
tend = "2023-01-01 00:01"

results = Fido.search(a.Time(tstart, tend), a.Instrument("HMI"), a.Physobs("intensity"))

###############################################################################
# Here we download the first file in the search results for example purpose, but this can be extended to more than one files.

files = Fido.fetch(results[0, 0])


###############################################################################
# Once the data is downloaded, we read the FITS files into a MapSequence object
# using SunPy's `sunpy.map.Map` function, which allows us to handle a sequence
# of solar images.

mc = sunpy.map.Map(files, sequence=True)

###############################################################################
# HMI maps are inverted, meaning that the solar north pole appears at the
# bottom of the image. To correct this, we rotate each map in the MapSequence
# using the `rotate` method with an order of 3. For detailed reason as to why, please refer this
# `example <https://docs.sunpy.org/en/stable/generated/gallery/map_transformations/upside_down_hmi.html#sphx-glr-generated-gallery-map-transformations-upside-down-hmi-py>`__.

cont_rotated = sunpy.map.MapSequence([m.rotate(order=3) for m in mc])

###############################################################################
# Plot the first map in the rotated MapSequence

fig = plt.figure()
ax = plt.subplot(projection=cont_rotated[0])
im = cont_rotated[0].plot(axes=ax, autoalign=True)

plt.show()

###############################################################################
# To reduce computational expense, we resample the continuum image to a lower
# resolution. This step ensures that running the algorithm on the full-resolution
# image is not overly computationally expensive.

cont_rotated_resample = cont_rotated[0].resample((1024, 1024) * u.pixel)

###############################################################################
# Next, we use the STARA function to detect sunspots in the resampled map.

segs = stara(cont_rotated_resample, limb_filter=10 * u.percent)

###############################################################################
# Finally, we plot the resampled map along with the detected sunspots. We create
# a new Matplotlib figure and subplot with the projection defined by the resampled
# map, plot the resampled map, and overlay contours of the detected sunspots using
# the `contour` method.

fig = plt.figure()
ax = plt.subplot(projection=cont_rotated_resample)
im = cont_rotated_resample.plot(axes=ax, autoalign=True)
ax.contour(segs, levels=0)

plt.show()

###############################################################################
# To provide a closer view of specific regions containing sunspots, we can zoom
# in on a rectangular area of interest within the plotted map. Here, we define
# the coordinates of the rectangle to zoom in on.
# For example, let's zoom in on a rectangle from (x0, y0) to (x1, y1)

x0, y0 = 100, 200
x1, y1 = 500, 500

fig = plt.figure()
ax = plt.subplot(projection=cont_rotated_resample)
im = cont_rotated_resample.plot(axes=ax, autoalign=True)
ax.contour(segs, levels=0)

ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)

plt.show()
