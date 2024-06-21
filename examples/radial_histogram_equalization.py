"""
=============================
Radial Histogram Equalization
=============================

This example applies the radial histogram equalizing filter (`sunkit_image.radial.rhef`) filter to a sunpy map.
"""

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sunpy.data.sample
import sunpy.map

import sunkit_image.enhance as enhance
import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins

#######################################################################################
# Let us use the sunpy sample data AIA image to showcase the RHE filter.

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

# Create radial segments (RHEF should use a dense grid)
radial_bin_edges = equally_spaced_bins(0, 2, aia_map.data.shape[0] // 2)
radial_bin_edges *= u.R_sun

rhef_map = radial.rhef(aia_map, radial_bin_edges)


#######################################################################################
# It seems that the native sunpy plot routine has a strong effect on the output,
# so we recommend using imshow.

# Plot the three maps in a single figure with one row and three columns
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex="all", sharey="all", subplot_kw={"projection": aia_map})

# Original AIA map.plot()
ax = axes[0]
aia_map.plot(axes=ax, clip_interval=(1, 99.99) * u.percent)
ax.set_title("Original AIA Map")

# RHEF map.plot()
ax = axes[1]
rhef_map.plot(axes=ax, clip_interval=(1, 99.99) * u.percent)
ax.set_title("RHE map.plot()")

# RHE imshow(map.data)
ax = axes[2]
ax.imshow(rhef_map.data, origin="lower", extent=None, cmap=plt.get_cmap("sdoaia171"))
ax.set_title("RHE imshow()")

plt.tight_layout()
plt.show()


#######################################################################################
# The RHEF has one free parameter that works in post processing to modulate the output.
# Here are some of the choices one could make.
# See the thesis (Gilly 2022) for details about upsilon.

# Define the list of upsilon pairs where the first number affects dark components and the second number affects bright ones
upsilon_list = [
    0.35,
    None,
    (0.1, 0.1),
    (0.5, 0.5),
    (0.8, 0.8),
]


#######################################################################################
# Call the plotting functions

# Create a figure with subplots for each upsilon pair plus the original map
fig, axs = plt.subplots(2, 3, figsize=(15, 10), sharey="all", sharex="all")
axs = axs.flatten()

# Extract the coordinate ranges from the meta information
x_coords = aia_map.meta["cdelt1"] * (aia_map.data.shape[1] // 2)
y_coords = aia_map.meta["cdelt2"] * (aia_map.data.shape[0] // 2)
extent = [-x_coords, x_coords, -y_coords, y_coords]

# Plot the original map
# Adjust the map data to avoid log of zero
sdata = aia_map.data
epsilon = 1e-2
data0 = np.log10(np.maximum(sdata - np.nanmin(sdata), epsilon)) ** 2
im0 = axs[0].imshow(data0, origin="lower", extent=extent, cmap=mpl.colormaps["sdoaia171"])
axs[0].set_title("Log10(data)^2")

# Loop through the upsilon_list and plot each filtered map
for i, upsilon in enumerate(upsilon_list):
    out_map = radial.rhef(aia_map, radial_bin_edges=radial_bin_edges, upsilon=upsilon, method="scipy")
    data = out_map.data
    im = axs[i + 1].imshow(data, origin="lower", extent=extent, cmap=mpl.colormaps["sdoaia171"])
    axs[i + 1].set_title(f"Upsilon = {upsilon}")

# Adjust layout
plt.tight_layout()
plt.show()

#######################################################################################
# Note that multiple filters can be used in a row to get the best images


mgn_map = enhance.mgn(aia_map)
rhef_mgn_map = radial.rhef(mgn_map, radial_bin_edges)
rhef_map = radial.rhef(aia_map, radial_bin_edges)

# Plot the three maps in a single figure with one row and three columns
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex="all", sharey="all", subplot_kw={"projection": aia_map})
axes = axes.flatten()


ax = axes[0]
ax.imshow(mgn_map.data, origin="lower", cmap=plt.get_cmap("sdoaia171"))
ax.set_title("MGN()")


ax = axes[1]
ax.imshow(rhef_mgn_map.data, origin="lower", cmap=plt.get_cmap("sdoaia171"))
ax.set_title("RHEF(MGN())")


ax = axes[3]
toplot = (rhef_map.data + rhef_mgn_map.data) / 2
ax.imshow(toplot, origin="lower", cmap=plt.get_cmap("sdoaia171"))
ax.set_title("(RHEF() + RHEF(MGN()))/2")


ax = axes[2]
ax.imshow(rhef_map.data, origin="lower", cmap=plt.get_cmap("sdoaia171"))
ax.set_title("RHEF()")

plt.tight_layout()
plt.show()
