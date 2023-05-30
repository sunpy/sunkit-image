"""
=================================== Coalignment using Template Matching
===================================

A common approach to coaligning a time series of images is to take a
representative template that contains the features you are interested in, and
match that to your images. The location of the best match tells you where the
template is in your image. The images are then shifted to the location of the
best match. This aligns your images to the position of the features in your
representative template.

This example demonstrates how to coalign maps in a `~sunpy.MapSequence` using
the :func:`~sunkit_image.coalignment.mapsequence_coalign_by_match_template`
function. The implementation of this functionality requires the installation
of the image processing library
`scikit-image <https://scikit-image.org/docs/stable/install.html>`__.
"""
import matplotlib.pyplot as plt

import sunpy.data.sample
from sunpy.map import Map

from sunkit_image import coalignment

###############################################################################
# Create a `~sunpy.map.MapSequence` using sample data.
mc = Map(
    [
        sunpy.data.sample.AIA_193_CUTOUT01_IMAGE,
        sunpy.data.sample.AIA_193_CUTOUT02_IMAGE,
        sunpy.data.sample.AIA_193_CUTOUT03_IMAGE,
        sunpy.data.sample.AIA_193_CUTOUT04_IMAGE,
        sunpy.data.sample.AIA_193_CUTOUT05_IMAGE,
    ],
    sequence=True,
)

###############################################################################
# Plot an animation of the `~sunpy.map.MapSequence` that we can compare with
# the coaligned MapSequence.
plt.figure()
anim = mc.plot()
plt.show()

###############################################################################
# To coalign the `~sunpy.map.MapSequence`, apply the
# :func:`~sunkit_image.coalignment.mapsequence_coalign_by_match_template`
# function.
coaligned = coalignment.mapsequence_coalign_by_match_template(mc)

###############################################################################
# This returns a new `~sunpy.map.MapSequence` coaligned to a template
# extracted from the center of the first map in the `~sunpy.map.MapSequence`,
# with the dimensions clipped as required.
#
# For a full list of options and functionality of the coalignment algorithm,
# see `~sunkit_image.coalignment.mapsequence_coalign_by_match_template`.
#
# Now, let's plot an animation of the coaligned MapSequence to compare with
# the original.
plt.figure()
anim_coalign = coaligned.plot()
plt.show()

###############################################################################
# If you just want to calculate the shifts required to compensate for solar
# rotation relative to the first map in the `~sunpy.map.MapSequence` without
# applying the shifts, use
# :func:`~sunkit_image.coalignment.calculate_match_template_shift`:
shifts = coalignment.calculate_match_template_shift(mc)

###############################################################################
# This is the function used to calculate the shifts in
# :func:`~sunkit_image.coalignment.mapsequence_coalign_by_match_template`.
# The shifts calculated here can be passed directly to the coalignment
# function.
coaligned = coalignment.mapsequence_coalign_by_match_template(mc, shift=shifts)
