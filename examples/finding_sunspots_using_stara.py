"""
============================
Finding sunspots using STARA
============================

This example demonstrates the use of Sunspot Tracking And Recognition
Algorithm (STARA) in detecting and plotting sunspots. More information
on the algorithm can be found in [this](https://arxiv.org/abs/1009.5884) paper.
If you wish to perform analysis over a large period of time we suggest to refer
[this](https://gitlab.com/wtbarnes/aia-on-pleiades/-/blob/master/notebooks/tidy/finding_sunspots.ipynb)
notebook implementation of the same algorithm using dask arrays.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.io._fits
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

from sunkit_image.stara import stara

###############################################################################
# Firstly, let's download HMI continuum data from the Virtual Solar Observatory.

query = Fido.search(a.Time("2023-01-01 00:00", "2023-01-01 00:01"), a.Instrument("HMI"), a.Physobs("intensity"))
# We only download the first file in the search results in this example.
file = Fido.fetch(query[0, 0])

###############################################################################
# Once the data is downloaded, we read the FITS files into a Map object
# using SunPy's `sunpy.map.Map` function.

hmi_map = sunpy.map.Map(file)

###############################################################################
# HMI maps are inverted, meaning that the solar north pole appears at the
# bottom of the image. To correct this, we rotate each map in the MapSequence
# using the ``rotate`` method with an order of 3. For detailed reason as to why, please refer this
# `example <https://docs.sunpy.org/en/stable/generated/gallery/map_transformations/upside_down_hmi.html#sphx-glr-generated-gallery-map-transformations-upside-down-hmi-py>`__.

cont_rotated = hmi_map.rotate(order=3)

###############################################################################
# Plot the rotated map.

fig = plt.figure()
ax = plt.subplot(projection=cont_rotated)
im = cont_rotated.plot(axes=ax, autoalign=True)

plt.show()

###############################################################################
# To reduce computational expense, we resample the continuum image to a lower
# resolution. This step ensures that running the algorithm on the full-resolution
# image is not overly computationally expensive.

cont_rotated_resample = cont_rotated.resample((1024, 1024) * u.pixel)

###############################################################################
# Next, we use the STARA function to detect sunspots in the resampled map.

segs = stara(cont_rotated_resample, limb_filter=10 * u.percent)

###############################################################################
# Finally, we plot the resampled map along with the detected sunspots. We create
# a new Matplotlib figure and subplot with the projection defined by the resampled
# map, plot the resampled map, and overlay contours of the detected sunspots using
# the ``contour`` method.

fig = plt.figure()
ax = plt.subplot(projection=cont_rotated_resample)
im = cont_rotated_resample.plot(axes=ax, autoalign=True)
ax.contour(segs, levels=0)

plt.show()

###############################################################################
# To focus on specific regions containing sunspots, we can create a submap,
# which is a smaller section of the original map. This allows us to zoom in
# on areas of interest. We define the coordinates of the rectangle to crop
# in pixel coordinates.

# Convert the pixel coordinates to world coordinates. This step is necessary
# because the submap method expects input in world coordinates.
bottom_left = cont_rotated_resample.pixel_to_world(240 * u.pix, 350 * u.pix)
top_right = cont_rotated_resample.pixel_to_world(310 * u.pix, 410 * u.pix)

# Create the submap using the world coordinates of the bottom left and top right corners.
hmi_submap = cont_rotated_resample.submap(bottom_left, top_right=top_right)
segs = stara(hmi_submap, limb_filter=10 * u.percent)

# Plot the submap along with the contours.
fig = plt.figure()
ax = plt.subplot(projection=hmi_submap)
im = hmi_submap.plot(axes=ax, autoalign=True)
ax.contour(segs, levels=0)

plt.show()
