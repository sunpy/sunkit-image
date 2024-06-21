"""
===================================
Radial Histogram Equalization
===================================

This example applies the radial histogram equalizing filter (`sunkit_image.radial.rhef`) filter to a sunpy map.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
import sunpy.data.sample
import sunpy.map

import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins
import sunkit_image.utils as utils

###########################################################################
# Sunpy's sample data contain a number of suitable FITS files for this purpose.

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

# The original image is plotted to showcase the difference.
fig = plt.figure()
ax = plt.subplot(projection=aia_map)
aia_map.plot(clip_interval=(1, 99.99) * u.percent)

###########################################################################
# Here we create the radial segments. Each segment created will be of
# equal dimensions radially. The distance between 0 solar radii and 2 solar radii
# is divided into 100 equal parts by the following two lines.

radial_bin_edges = equally_spaced_bins(0, 2, aia_map.data.shape[0])
radial_bin_edges *= u.R_sun

# The rhef filter is applied after it.
out1 = radial.rhef(aia_map, radial_bin_edges)

# The RHE filtered map is plotted.
fig = plt.figure()
ax = plt.subplot(projection=out1)
out1.plot()
plt.show()

# The RHEF has one free parameter that works in post processing to modulate the output.

radial_bin_edges = utils.equally_spaced_bins(0, 2, aia_map.data.shape[1])
radial_bin_edges *= u.R_sun

# Define the list of upsilon pairs
upsilon_list = [
    0.35,
    None,
    (0.1, 0.1),
    (0.5, 0.5),
    (0.8, 0.8),
]

import numpy as np

# Call the plotting functions
# Adjust the map data to avoid log of zero
sdata= aia_map.data

# Small constant to avoid log of zero
epsilon = 1e-2

# Adjust the data to avoid log of zero
data0 = np.log10(np.maximum(sdata - np.nanmin(sdata), epsilon)) ** 2

# Extract the coordinate ranges from the meta information
x_coords = aia_map.meta['cdelt1'] * (aia_map.data.shape[1] // 2)
y_coords = aia_map.meta['cdelt2'] * (aia_map.data.shape[0] // 2)
extent = [-x_coords, x_coords, -y_coords, y_coords]

# Create a figure with subplots for each upsilon pair plus the original map
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey='all', sharex='all')
axs = axs.flatten()

# Plot the original map
im0 = axs[0].imshow(data0, origin='lower', extent=extent, cmap=matplotlib.colormaps['sdoaia171'])
axs[0].set_title("Log10(data)^2")

# Loop through the upsilon_list and plot each filtered map
for i, upsilon in enumerate(upsilon_list):
    out_map = radial.rhef(aia_map, radial_bin_edges=radial_bin_edges, upsilon=upsilon, method="scipy")
    data = out_map.data
    im = axs[i + 1].imshow(data, origin='lower', extent=extent, cmap=matplotlib.colormaps['sdoaia171'])
    axs[i + 1].set_title(f"Upsilon = {upsilon}")

# Adjust layout
plt.tight_layout()

plt.show()