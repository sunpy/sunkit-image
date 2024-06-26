"""
=============================
Radial Histogram Equalization
=============================

This example applies the Radial Histogram Equalizing Filter (`sunkit_image.radial.rhef`) to a sunpy map.
"""

import astropy.units as u
import matplotlib.pyplot as plt
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

rhef_map = radial.rhef(aia_map, radial_bin_edges, vignette=True)


# Plot the 2 maps in a single figure with one row and three columns
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex="all", sharey="all", subplot_kw={"projection": aia_map})

# Original AIA map.plot()
ax = axes[0]
aia_map.plot(axes=ax, clip_interval=(1, 99.99) * u.percent)
ax.set_title("Original AIA Map")

# RHEF map.plot()
ax = axes[1]
rhef_map.plot(axes=ax, norm=None)  # This norm=None is very important for visualization
ax.set_title("RHE Filtered Map")

plt.show()


#######################################################################################
# The RHEF has one free parameter that works in post processing to modulate the output.
# Here are some of the choices one could make.
# See the thesis (Gilly 2022) Eq 4.15 for details about upsilon: https://www.proquest.com/docview/2759080511

# Define the list of upsilon pairs where the first number affects dark components and the second number affects bright ones
upsilon_list = [
    0.35,
    None,
    (0.1, 0.1),
    (0.5, 0.5),
    (0.8, 0.8),
]

# Create a figure with subplots for each upsilon pair plus the original map
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex="all", sharey="all", subplot_kw={"projection": aia_map})
axs = axes.flatten()

aia_map.plot(axes=axs[0], clip_interval=(1, 99.99) * u.percent)
axs[0].set_title("Original AIA Map")

# # Loop through the upsilon_list and plot each filtered map
for i, upsilon in enumerate(upsilon_list):
    out_map = radial.rhef(aia_map, upsilon=upsilon, vignette=True, method="scipy")
    out_map.plot(axes=axs[i + 1], norm=None)
    axs[i + 1].set_title(f"Upsilon = {upsilon}")

# Adjust layout
plt.show()

#######################################################################################
# Note that multiple filters can be used in a row to get the best images
# Here, we will use both MGN and WOW, then apply RHEF after.
mgn_map = enhance.mgn(aia_map)
wow_map = enhance.wow(aia_map)

rhef_map = radial.rhef(aia_map, vignette=True)

rhef_mgn_map = radial.rhef(mgn_map, vignette=True)
rhef_wow_map = radial.rhef(wow_map, vignette=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex="all", sharey="all", subplot_kw={"projection": aia_map})
axes = axes.flatten()

ax = axes[1]
mgn_map.plot(axes=ax, norm=None)
ax.set_title("MGN(smap)")

ax = axes[4]
rhef_mgn_map.plot(axes=ax, norm=None)
ax.set_title("RHEF( MGN(smap) )")

ax = axes[0]
rhef_map.plot(axes=ax, norm=None)
ax.set_title("RHEF(smap)")

ax = axes[3]
toplot = (rhef_map.data + rhef_mgn_map.data) / 2
combo_map = sunpy.map.Map(toplot, rhef_map.meta)
combo_map.plot(axes=ax, norm=None)
ax.set_title("AVG( RHEF(smap), RHEF(MGN(smap) )")

ax = axes[2]
wow_map.plot(axes=ax, norm=None)
ax.set_title("WOW(smap)")

ax = axes[5]
rhef_wow_map.plot(axes=ax, norm=None)
ax.set_title("RHEF( WOW(smap) )")

plt.show()
