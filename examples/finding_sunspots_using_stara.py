"""
============================
Finding sunspots using STARA
============================

This example demonstrates the use of Sunspot Tracking And Recognition
Algorithm (STARA) in detecting and plotting sunspots.

More information on the algorithm can be found in `this paper. <https://doi.org/10.1017/S1743921311014992>`__

If you wish to perform analysis over a large period of time we suggest to refer
`this <https://gitlab.com/wtbarnes/aia-on-pleiades/-/blob/master/notebooks/tidy/finding_sunspots.ipynb>`__
notebook implementation of the same algorithm using dask arrays.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
from skimage.measure import label, regionprops_table

import astropy.units as u
from astropy.table import QTable
from astropy.time import Time

import sunpy.io._fits
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a

from sunkit_image.stara import stara

###############################################################################
# Firstly, let's download HMI continuum data from the Virtual Solar Observatory (VSO).

query = Fido.search(a.Time("2023-01-01 00:00", "2023-01-01 00:01"), a.Instrument("HMI"), a.Physobs("intensity"))
hmi_file = Fido.fetch(query)

###############################################################################
# Once the data is downloaded, we read the FITS file using`sunpy.map.Map`.
#
# To reduce computational expense, we resample the continuum image to a lower
# resolution as this is run on a small cloud machine.
#
# HMI images are inverted, meaning that the solar north pole appears at the
# bottom of the image. To correct this, we rotate each map in the MapSequence
# using the ``rotate`` method with an order of 3.
#
# We will combine these into one step.

hmi_map = sunpy.map.Map(hmi_file).resample((1024, 1024) * u.pixel).rotate(order=3)

###############################################################################
# Next, we use the :func:`~sunkit_image.stara.stara` function to detect sunspots in data.

stara_segments = stara(hmi_map, limb_filter=10 * u.percent)

###############################################################################
# Now we will plot the detected contours from STARA on the HMI data.

fig = plt.figure()
ax = fig.add_subplot(111, projection=hmi_map)
hmi_map.plot(axes=ax)
ax.contour(stara_segments, levels=0)

fig.tight_layout()

###############################################################################
# To focus on specific regions containing sunspots, we can create a submap,
# which is a smaller section of the original map. This allows us to zoom in
# on areas of interest. We define the coordinates of the rectangle to crop
# in pixel coordinates.

bottom_left = hmi_map.pixel_to_world(240 * u.pix, 350 * u.pix)
top_right = hmi_map.pixel_to_world(310 * u.pix, 410 * u.pix)

hmi_submap = hmi_map.submap(bottom_left, top_right=top_right)
stara_segments = stara(hmi_submap, limb_filter=10 * u.percent)

###############################################################################
# We can further enhance our analysis by extracting key properties from the
# segmented image and organizing them into a structured table.
# First, a labeled image is created where each connected component (a sunspot)
# is assigned a unique label.

labelled = label(stara_segments)

# Extract properties of the labeled regions
regions = regionprops_table(
    labelled,
    hmi_submap.data,
    properties=[
        "label",  # Unique for each sunspot
        "centroid",  # Centroid coordinates (center of mass)
        "area",  # Total area (number of pixels)
        "min_intensity",
    ],
)
# We will add a new column named "obstime" is added to the table, which contains
# the observation date for each sunspot.
regions["obstime"] = Time([hmi_submap.date] * regions["label"].size)
# The pixel coordinates of sunspot centroids are converted to world coordinates
# (solar longitude and latitude) in Heliographic Stonyhurst (HGS).
regions["center_coord"] = hmi_submap.pixel_to_world(
    regions["centroid-0"] * u.pix,
    regions["centroid-1"] * u.pix,
).heliographic_stonyhurst
print(QTable(regions))

###############################################################################
# Further we could also plot a map with the corresponding center coordinates
# marked and their number.

# Extract centroid coordinates.
centroids_x = regions["centroid-1"]
centroids_y = regions["centroid-0"]

fig = plt.figure()
ax = fig.add_subplot(111, projection=hmi_submap)
hmi_submap.plot(axes=ax)
ax.contour(stara_segments, levels=0)
ax.scatter(centroids_x, centroids_y, color="red", marker="o", s=30, label="Centroids")
# Label each centroid with its corresponding sunspot label for better identification.
for i, labels in enumerate(regions["label"]):
    ax.text(centroids_x[i], centroids_y[i], f"{labels}", color="yellow", fontsize=16)
fig.tight_layout()

plt.show()
