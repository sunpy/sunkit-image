"""
=====================
Coronal Loops Tracing
=====================

This example applies traces out the coronal loops in a FITS file
using `~sunkit_image.trace.occult2`. In this example we will use the settings
and the data from Markus Aschwanden's tutorial on his IDL implementation of
the ``OCCULT2`` algorithm, which can be found `here <http://www.lmsal.com/~aschwand/software/tracing/tracing_tutorial1.html>`__.
The aim of this example is to demonstrate that `~sunkit_image.trace.occult2` provides similar
results as the IDL version.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import astropy
import numpy as np

import sunkit_image.trace as trace

###########################################################################
# We will be using `astropy.io.fits.getdata` to read the FITS file from the tutorial website.
image = astropy.io.fits.getdata("http://www.lmsal.com/~aschwand/software/tracing/TRACE_19980519.fits", ignore_missing_end=True)

# The original image is plotted to showcase the difference.
plt.imshow(image, cmap="Spectral", origin="lower")

###########################################################################
# Now the loop tracing will begin. We will be using the same set of parameters
# as in the IDL tutorial.
# * The lowpass filter boxcar filter size ``nsm1`` is taken to be 3.
# * The minimum radius of curvature at any point in the loop ``rmin`` is 30 pixels.
# * The length of the smallest loop to be detected ``lmin`` is 25 pixels.
# * The maximum number of structures to be examined ``nstruc`` is 1000.
# * The maximum number of loops to be detected ``nloop`` is 1000.
# * The number of extra points in the loop below noise level to terminate a loop tracing ``ngap`` is 0.
# * The base flux and median flux ratio ``qthresh1`` is 0.0
# * The noise threshold in the image with repect to median flux ``qthresh2`` is 3.0 
loops = trace.occult2(image, nsm1=3, rmin=30, lmin=25, nstruc=1000, nloop=1000, 
                                  ngap=0, qthresh1=0.0, qthresh2=3.0, file=True)

###############################################################################
# `~sunkit_image.trace.occult2` returns a list of all loop where each element
# is itself a list of points containg ``x`` and ``y`` coordinates for each point.
# Now we will plot all the loops detected.

fig = plt.figure()
for loop in loops:

    # We collect all the ``x`` and ``y`` coordinates in seperate lists for plotting.
    x = []
    y = []
    for points in loop:
        x.append(points[0])
        y.append(points[1])
    
    plt.plot(x, y, 'b')
    
#################################################################################
# We also plot the original image to overlay the loops over it.

plt.imshow(image, cmap="Spectral", origin="lower")

plt.show()
