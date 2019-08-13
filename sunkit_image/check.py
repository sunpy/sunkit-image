import numpy as np
import sunpy.io
import matplotlib.pyplot as plt
import sunkit_image.trace as trace
import astropy

# def test_image():
#     ima = np.zeros((11, 11), dtype=float)
#     ima[:, 5] = 1.0
#     ima[3, 6] = 1.0
#     ima[4, 7] = 1.0
#     ima[5, 8] = 1.0
#     ima[4, 9] = 1.0
#     ima[3, 10] = 1.0
#     return ima

# image = test_image()
image = astropy.io.fits.getdata("http://www.lmsal.com/~aschwand/software/tracing/TRACE_19980519.fits", ignore_missing_end=True)

loops = trace.occult2(image, nsm1=3, rmin=30, lmin=25, nstruc=1000, nloop=1000, ngap=0, qthresh1=0.0, qthresh2=3.0, file=True)
