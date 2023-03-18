"""
====================================
Persistence transform for a sequence
====================================

Persistence transform is a simple image processing technique that is useful for the visualization
and depiction of gradually evolving structures.
This example illustrates how to perform a persistence transform.

.. note::

   This example requires `mpl_animators` to be installed.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

import sunpy.map
from sunpy.data.sample import AIA_193_CUTOUT01_IMAGE, AIA_193_CUTOUT02_IMAGE, AIA_193_CUTOUT03_IMAGE

###############################################################################
# We will create a `~sunpy.map.MapSequence` using the `sunpy` sample data.

aiamapseq = sunpy.map.Map(AIA_193_CUTOUT01_IMAGE, AIA_193_CUTOUT02_IMAGE, AIA_193_CUTOUT03_IMAGE)

###############################################################################
# For a data set consisting of N images with intensity values I(x,y,t),
# the Persistence Map Pn is a function of several arguments,
# namely intensity, location and time:
# Pn(x,y,tn)=Q(I(x,y,t≤tn))

persistence_maps = []
for i, map_i in enumerate(aiamapseq[1:]):
    per = np.array([aiamapseq[n].data for n in [i, i + 1]]).max(axis=0)
    smap = sunpy.map.Map(per, map_i.meta)
    smap.plot_settings["cmap"] = cm.get_cmap("Greys_r")
    smap.plot_settings["norm"] = colors.LogNorm(100, smap.max())
    persistence_maps.append(smap)

###############################################################################
# This plots the original mapseq
for i in aiamapseq:
    i.peek()

###############################################################################
# Now we will plot the original sequence and the new sequence side by side so you can see the difference.

result_mapseq = sunpy.map.MapSequence(persistence_maps)
result_mapseq.peek()
plt.show()
