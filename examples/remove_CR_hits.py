"""
========================
Removing Cosmic Ray Hits
========================

This example illustrates how to remove cosmic ray hits from a LASCO C2 image
using `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__.
Astroscrappy is a separate Python package and can be installed separately using ``pip`` or ``conda``.
"""

import matplotlib.pyplot as plt

from sunpy.map import Map

import astroscrappy

###############################################################################
# First, we will work with the FITS files.
# These imports are necessary to download the FITS file based on the instrument.
# For more details on how to download a particular LASCO FITS file as a map see
# Sunpy's example `Downloading and plotting LASCO C3 data <https://docs.sunpy.org/en/stable/generated/gallery/acquiring_data/skip_downloading_lascoC3.html>`__.
# To make this example work you need to have SunPy with all the "net" dependencies installed.
from sunpy.net import Fido, attrs as a
from sunpy.io.file_tools import read_file

###############################################################################
# In order to download the required FITS file, we use
# `Fido <sunpy.net.fido_factory.UnifiedDownloaderFactory>`, a downloader client.
# We define two search variables:
# a timerange and the instrument.
timerange = a.Time("2000/11/09 00:06", "2000/11/09 00:07")
instrument = a.Instrument("LASCO")
detector = a.Detector("C2")
result = Fido.search(timerange, instrument)

downloaded_files = Fido.fetch(result[0])
data, header = read_file(downloaded_files[0])[0]

# Add the missing meta information to the header
header["CUNIT1"] = "arcsec"
header["CUNIT2"] = "arcsec"

###############################################################################
# With this fix we can load it into a map and plot the results.
lascomap1 = Map(data, header)
fig1 = plt.figure()
lascomap1.plot()

###############################################################################
# Now we will call the `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__
# to remove the cosmic ray hits. This algorithm will perform well with both high and low
# noise levels in the FITS file.
# The function takes a numpy.ndarray as input so we only pass the data part of
# the map. This particular image has lots of high intensity cosmic ray hits which
# cannot be effectively removed by using the default set of parameters.
# So we reduce ``sigclip``, the Laplacian to noise ratio from 4.5 to 2 to mark more hits.
# We also reduce ``objlim``, contrast between the Laplacian image and the fine structed image
# to clean the high intensity bright cosmic ray hits.
# We also modify the ``readnoise`` parameters to obtain better results.

mask, clean = astroscrappy.detect_cosmics(lascomap1.data, sigclip=2, objlim=2, readnoise=4, verbose=True)
# This returns two values - mask is a boolean array depicting whether there is
# a cosmic ray hit at that pixel, clean is the cleaned image after removing those
# hits.

###############################################################################
# We can now plot the cleaned image after making a `sunpy.map.GenericMap`.
clean_map1 = Map(clean, lascomap1.meta)

fig2 = plt.figure()
clean_map1.plot()

# Show all the plots
plt.show()
