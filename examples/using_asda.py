"""
================
Detecting Swirls
================

This example showcases how to use Automated Swirl Detection Algorithm (ASDA) to
detect swirls in velocity fields.
"""

import matplotlib.pyplot as plt
import numpy as np

from sunkit_image.asda import Asda
from sunkit_image.data.test import get_test_filepath

###########################################################################
# This examples demonstrates find swirls in a 2D velocity flow field.
# We will use precomputed a flow field from our test data.
# First thing is to load the velocity field and data.

vxvy = np.load(get_test_filepath("asda_vxvy.npz"))
vx = vxvy["vx"]
vy = vxvy["vy"]
data = vxvy["data"]

###########################################################################
# Now we will call `~sunkit_image.asda.Asda` to perform swirl detection

# Initialise class
# TODO Explain Factor
lo = Asda(vx, vy, factor=1)
# Gamma
# TODO Explain Gamma
gamma = lo.gamma_values()
# Determine Swirls
# TODO Explain This
center_edge = lo.center_edge()
# Properties of Swirls
# TODO Explain This
# DO I even want this?
# Useful in a table maybe?
ve, vr, vc, ia = lo.vortex_property(image=data)

###########################################################################
# Finally we will visualize the results.

# TODO What about a swirl map?
fig, ax = plt.subplots()
ax.imshow(gamma[..., 0], origin="lower")
ax.set_title(r"$\Gamma_1$")
ax.set(xlabel="x", ylabel="y")
plt.show()

fig, ax = plt.subplots()
ax.imshow(gamma[..., 1], origin="lower")
ax.set_title(r"$\Gamma_2$")
ax.set(xlabel="x", ylabel="y")
plt.show()
