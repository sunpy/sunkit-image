"""
========================================
Detecting Swirls in the Solar Atmosphere
========================================

This example demonstrates the use of Automated Swirl Detection Algorithm (ASDA) in detecting and
plotting swirls (vortices) in a 2D velocity flow field.

More information on the algorithm can be found in `this paper. <https://doi.org/10.3847/1538-4357/aabd34>`__

"""
# sphinx_gallery_thumbnail_number = 4 # NOQA: ERA001

import matplotlib.pyplot as plt
import numpy as np

from sunkit_image.asda import calculate_gamma_values, get_vortex_edges, get_vortex_properties
from sunkit_image.data.test import get_test_filepath

###########################################################################
# This example demonstrates how to find swirls (vortices) in a 2D velocity flow field.
# We will use precomputed flow field data from our test dataset.
# First, we load the velocity field and additional data.

vxvy = np.load(get_test_filepath("asda_vxvy.npz"))
vx = vxvy["vx"]
vy = vxvy["vy"]
data = vxvy["data"]

###########################################################################
# Now we will perform swirl detection using the methods provided in the `~sunkit_image.asda` module.
# The first step is to calculate the Gamma values. Gamma1 (Γ1) is useful for identifying
# vortex centers, while Gamma2 (Γ2) helps in detecting the edges of vortices.
# These values are calculated based on the method proposed by `Graftieaux et al. (2001) <https://doi.org/10.1088/0957-0233/12/9/307>`__
# and are used to quantify the swirling strength and structure of the flow field.
# To enhance the detection of smaller swirls and improve the accuracy in identifying
# vortex boundaries, a factor is introduced that magnifies the original data. This
# magnification aids in enhancing the resolution of the velocity field, allowing for
# more precise detection of sub-grid vortex centers and boundaries. By default, the
# factor is set to 1, but it can be adjusted based on the resolution of the data.

gamma = calculate_gamma_values(vx, vy, factor=1)

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
# Finally, we visualize the results. We will plot the Gamma1 and Gamma2 values,
# which highlight the vortex centers and edges respectively.

fig, ax = plt.subplots()
ax.imshow(gamma[..., 0], origin="lower")
ax.set_title(r"$\Gamma_1$")
ax.set(xlabel="x", ylabel="y")

fig, ax = plt.subplots()
ax.imshow(gamma[..., 1], origin="lower")
ax.set_title(r"$\Gamma_2$")
ax.set(xlabel="x", ylabel="y")

###########################################################################
# Furthermore, we can create a swirl map visualization.
# Generating a swirl map is a crucial step in analyzing the dynamics of fluid flow,
# particularly in identifying and understanding the spatial distribution, size, and
# characteristics of swirls within the velocity field. This visualization not only
# aids in the qualitative assessment of the flow patterns but also provides insights
# into the underlying physical processes driving the formation and evolution of these swirls.

fig, ax = plt.subplots()
ax.imshow(data, origin="lower", cmap="gray")
ax.quiver(np.arange(vx.shape[1]), np.arange(vx.shape[0]), vx, vy, scale=50, color="lightgreen")

# Mark and number swirl centers
centers = np.array(center_edge["center"])
for i, center in enumerate(centers):
    ax.plot(center[0], center[1], "bo")
    ax.text(center[0], center[1], str(i), color="white", ha="right", va="bottom")

# Overlay swirl edges
for edge in center_edge["edge"]:
    edge = np.array(edge)
    ax.plot(edge[:, 0], edge[:, 1], "b--")

ax.set_title("Swirl Map with Velocity Field")
ax.set(xlabel="x", ylabel="y")

###########################################################################
# Now we will magnify a specific region of interest (ROI) of the swirl map and overlay streamlines
# to visualize the flow patterns around the identified swirls.

# Define the region of interest (ROI) in pixel coordinates
x_min, x_max = 1, 100
y_min, y_max = 1, 100

fig, ax = plt.subplots()
ax.imshow(data[y_min:y_max, x_min:x_max], origin="lower", cmap="gray")

# Overlay streamlines
Y, X = np.mgrid[y_min:y_max, x_min:x_max]
ax.streamplot(X, Y, vx[y_min:y_max, x_min:x_max], vy[y_min:y_max, x_min:x_max], color="green")

# Mark and number swirl centers within ROI
roi_centers = centers[
    (centers[:, 0] >= x_min) & (centers[:, 0] < x_max) & (centers[:, 1] >= y_min) & (centers[:, 1] < y_max)
]
for center in roi_centers:
    ax.plot(center[0] - x_min, center[1] - y_min, "ro")

# Overlay swirl edges within ROI
for edge in center_edge["edge"]:
    edge = np.array(edge)
    edge_in_roi = edge[(edge[:, 0] >= x_min) & (edge[:, 0] < x_max) & (edge[:, 1] >= y_min) & (edge[:, 1] < y_max)]
    if edge_in_roi.size > 0:
        ax.plot(edge_in_roi[:, 0] - x_min, edge_in_roi[:, 1] - y_min, "r--")

ax.set_title("Magnified Swirl Map Region with Streamlines")
ax.set(xlabel="x", ylabel="y")
plt.show()
