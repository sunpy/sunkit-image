"""
========================================
Detecting Swirls in the Solar Atmosphere
========================================

This example demonstrates the use of Automated Swirl Detection Algorithm (ASDA) in detecting and plotting swirls (vortices) in a 2D velocity flow field.

`More information on the algorithm can be found in the original paper. <https://doi.org/10.3847/1538-4357/aabd34>`__

Unfortunately, currently ASDA within sunkit-image only works on arrays.
"""
# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from sunkit_image.asda import calculate_gamma_values, get_vortex_edges, get_vortex_properties
from sunkit_image.data.test import get_test_filepath

###########################################################################
# This example demonstrates how to find swirls (vortices) in a 2D velocity flow field.
#
# Ideally you will want to calculate the velocity field from your data, but for this example
# we will use precomputed flow field data from our test dataset.
#
# `pyflct <https://pyflct.readthedocs.io/en/latest/>`__ is a good tool to calculate the velocity field from your data.

vxvy = np.load(get_test_filepath("asda_vxvy.npz"))
# This is the original data used to calculate the velocity field
data = vxvy["data"]
# These are the velocity components in the x and y directions
vx = vxvy["vx"]
vy = vxvy["vy"]

###########################################################################
# Before we proceed with swirl detection, let's understand data by
# visualizing the velocity magnitude.

# Calculate velocity magnitude
velocity_magnitude = np.sqrt(vx**2 + vy**2)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

im = ax.imshow(velocity_magnitude, origin="lower", cmap="viridis")
ax.set_title("Velocity Magnitude")
cax = make_axes_locatable(ax).append_axes("bottom", size="5%", pad="5%")
cbar = fig.colorbar(im, ax=cax, orientation="horizontal")
cbar.set_label("Velocity (m/s)")

###########################################################################
# Now we will perform swirl detection using the methods provided by `~sunkit_image.asda`.
#
# The first step is to calculate the Gamma values. Gamma1 (Γ1) is useful for identifying
# vortex centers, while Gamma2 (Γ2) helps in detecting the edges of vortices.
# These values are calculated based on the method proposed by `Graftieaux et al. (2001) <https://doi.org/10.1088/0957-0233/12/9/307>`__
# and are used to quantify the swirling strength and structure of the flow field.
#
# To enhance the detection of smaller swirls and improve the accuracy in identifying
# vortex boundaries, a factor is introduced that magnifies the original data. This
# magnification aids in enhancing the resolution of the velocity field, allowing for
# more precise detection of sub-grid vortex centers and boundaries. By default, the
# factor is set to 1, but it can be adjusted based on the resolution of the data.

gamma = calculate_gamma_values(vx, vy)

###########################################################################
# Next, we identify the edges and centers of the swirls using the calculated Gamma values.
# The :func:`~sunkit_image.asda.get_vortex_edges` function processes the Gamma2 values
# to locate the boundaries of vortices and uses Gamma1 to pinpoint their centers.

center_edge = get_vortex_edges(gamma)

###########################################################################
# We can also determine various properties of the identified vortices, such as their
# expanding speed (ve), rotational speed (vr), center velocity (vc), and average
# observational values (ia). This information can be useful for detailed analysis
# and is calculated using the :func:`~sunkit_image.asda.get_vortex_properties` function.

ve, vr, vc, ia = get_vortex_properties(vx, vy, center_edge, image=data)

###########################################################################
# Now we will plot the Gamma1 and Gamma2 values, which highlight the vortex
# centers and edges respectively.

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax.imshow(gamma[..., 0], origin="lower")
ax.set_title(r"$\Gamma_1$")
ax.set(ylabel="y")
ax.set_xticklabels([])

ax2.imshow(gamma[..., 1], origin="lower")
ax2.set_title(r"$\Gamma_2$")
ax2.set(xlabel="x", ylabel="y")

fig.tight_layout()

###########################################################################
# Finally, we can create a swirl map visualization with streamlines.

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

ax.imshow(data, origin="lower", cmap="gray")

# Overlay streamlines
Y, X = np.mgrid[0:512, 0:1024]
ax.streamplot(X, Y, vx, vy, color="green")

# Mark and number swirl centers
centers = np.array(center_edge["center"])
for i, center in enumerate(centers):
    ax.plot(center[0], center[1], "bo")
    ax.text(center[0], center[1], str(i), color="red", ha="right", va="bottom")

# Overlay swirl edges
for edge in center_edge["edge"]:
    edge = np.array(edge)
    ax.plot(edge[:, 0], edge[:, 1], "b--")

ax.set_title("Swirl Map Region with Streamlines")
ax.set(xlabel="x", ylabel="y")
fig.tight_layout()

plt.show()
