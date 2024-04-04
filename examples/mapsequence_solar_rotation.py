"""
===============================
Compensating for Solar Rotation
===============================

Often a set of solar image data consists of fixing the pointing of a field of
view for some time and observing. Features on the Sun will differentially
rotate depending on latitude, with features at the equator moving faster than
features at the poles.

In this example, the process of shifting images in a `~sunpy.map.MapSequence`
to account for the differential rotation of the Sun is demonstrated using the
:func:`~sunkit_image.coalignment.mapsequence_coalign_by_rotation` function.
"""

import matplotlib.pyplot as plt
import sunpy.data.sample
from sunpy.map import Map

from sunkit_image import coalignment

###############################################################################
# First, create a `~sunpy.map.MapSequence` with sample data.

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
# Let's plot the MapSequence so we can later compare it with the shifted
# result.

plt.figure()
anim = mc.plot()

###############################################################################
# The :func:`~sunkit_image.coalignment.mapsequence_coalign_by_rotation`
# function can be applied to the Map Sequence

derotated = coalignment.mapsequence_coalign_by_rotation(mc)

###############################################################################
# By default, the de-rotation shifts are calculated relative to the first map
# in the `~sunpy.map.MapSequence`.
# This function does not differentially rotate the image (see
# `Differentially rotating a map <https://docs.sunpy.org/en/stable/generated/gallery/differential_rotation/reprojected_map.html>`__
# for an example). It is useful for de-rotating images when the effects of
# differential rotation in the `~sunpy.map.MapSequence` can be ignored.
#
# See the docstring of
# :func:`~sunkit_image.coalignment.mapsequence_coalign_by_rotation`
# for more features of the function.
#
# To check that the applied shifts were reasonable, plot an animation of the
# shifted MapSequence to compare with the original plot above.

plt.figure()
anim_derotate = derotated.plot()

###############################################################################
# The de-rotation shifts used in the above function can be calculated without
# applying them using the
# :func:`~sunkit_image.coalignment.calculate_solar_rotate_shift` function.

shifts = coalignment.calculate_solar_rotate_shift(mc)

###############################################################################
# The calculated shifts can be passed as an argument to
# :func:`~sunkit_image.coalignment.mapsequence_coalign_by_rotation`.
derotate_shifts = coalignment.mapsequence_coalign_by_match_template(mc, shift=shifts)

plt.figure()
anim_derotate_shifts = derotate_shifts.plot()

plt.show()
