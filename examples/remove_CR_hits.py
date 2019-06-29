"""
========================
Removing Cosmic Ray Hits
========================

This example illustrates how to remove cosmic ray hits from a LASCO C2 image (using ``FITS`` and ``jp2``)
using `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__. Astroscrappy is a separate Python package and can be
installed separately using ``pip`` or ``conda``.
"""
# sphinx_gallery_thumbnail_number = 5

import matplotlib.pyplot as plt

from sunpy.map import Map

import astroscrappy

###############################################################################
# First, we will work with the FITS files.
# These imports are necessary to download the FITS file based on the instrument.
# For more details on how to download a particular LASCO FITS file as a map see
# Sunpy's example `Downloading and plotting LASCO C3 data <https://docs.sunpy.org/en/stable/generated/gallery/acquiring_data/skip_downloading_lascoC3.html>`__.
# To make this example work you need to have `Sunpy` with all the `net` dependencies also installed.
from sunpy.net import Fido, attrs as a
from sunpy.io.file_tools import read_file

###############################################################################
# We will also work with ``jp2`` images. So to download these files we will use
# Sunpy's Helioviewer.org client. For more information about the helioviewer client,
# see sunpy example `Querying Helioviewer.org with SunPy <https://docs.sunpy.org/en/stable/guide/acquiring_data/helioviewer.html>`__.
from sunpy.net.helioviewer import HelioviewerClient

###############################################################################
# In order to download the required ``FITS`` file, we use
# `Fido <sunpy.net.fido_factory.UnifiedDownloaderFactory>`, a downloader client.
# We define two search variables:
# a timerange and the instrument.
timerange = a.Time("2000/11/09 00:26", "2000/11/09 00:27")
instrument = a.Instrument("LASCO")
detector = a.Detector("C2")
result = Fido.search(timerange, instrument)

downloaded_files = Fido.fetch(result[0])
data, header = read_file(downloaded_files[1])[0]

# Add the missing meta information to the header
header["CUNIT1"] = "arcsec"
header["CUNIT2"] = "arcsec"

###############################################################################
# With this fix we can load it into a map and plot the results.
lascomap1 = Map(data, header)
fig1 = plt.figure()
lascomap1.plot()

###############################################################################
# Now we will call the `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__ to remove the cosmic ray
# hits. This algorithm will perform well with both high intensity and low intensity
# noise levels in the ``FTIS`` file.

# The function takes a `numpy.ndarray` as input so we only pass the data part of
# the map.
mask, clean = astroscrappy.detect_cosmics(lascomap1.data)
# This returns two values - `mask` is a boolean array depicting whether their is
# a comic ray hit at that pixel, `clean` is the cleaned image after removing those
# hits.

###############################################################################
# We can now plot the cleaned image after making a `sunpy.map.GenericMap`.
clean_map1 = Map(clean, lascomap1.meta)

fig2 = plt.figure()
clean_map1.plot()

###############################################################################
# The above portion explained how to use `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__ when working
# with ``FITS`` files. Now, we will see how can we remove cosmic ray hits in a ``jp2``
# image.

# First, we will create a HelioviewerClient
hv = HelioviewerClient()

# This will download the ``jp2`` image based on the date of the observation and the
# instruments used. This is a ``jp2`` image with a low level of noise intensity.
file = hv.download_jp2(
    "2003/04/16", observatory="SOHO", instrument="LASCO", measurement="C2", source_id=4
)

###############################################################################
# We can load the downloaded file into a `sunpy.map.GenericMap` and plot the
# results.
lascomap2 = Map(file)

fig3 = plt.figure()
lascomap2.plot()

################################################################################
# Now we will again call the `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__. It is to be noted that
# this algorithm may not produce expected results on high intensity noisy ``jp2``
# images. Although in our observations it worked satisfactorily for low intensity
# noise levels in a ``jp2`` image.

# Here, we first show results for a low noise level ``jp2`` image.
# The function takes a `numpy.ndarray` as input so we only pass the data part of
# the map.
mask, clean = astroscrappy.detect_cosmics(lascomap2.data)

###############################################################################
# We can now plot the cleaned image after making a `sunpy.map.GenericMap`.
clean_map2 = Map(clean, lascomap2.meta)

fig4 = plt.figure()
clean_map2.plot()

###############################################################################
# Now we will take a high intensity noisy ``jp2`` image to evaluate the results.

jp2_file = hv.download_jp2(
    "2000/11/09", observatory="SOHO", instrument="LASCO", measurement="C2", source_id=4
)

##############################################################################
# We can see the image by a making a `sunpy.map.GenericMap` and plotting the
# results.

lascomap3 = Map(jp2_file)

fig5 = plt.figure()
lascomap3.plot()

###############################################################################
# We now try to elimimate the cosmic ray hits.

# The function takes a `numpy.ndarray` as input so we only pass the data part of
# the map.
mask, clean = astroscrappy.detect_cosmics(lascomap3.data)

###############################################################################
# The final clean image can be plotted as a map.

clean_map3 = Map(clean, lascomap3.meta)

fig6 = plt.figure()
clean_map3.plot()

###############################################################################
# We can plot all the results at the end.
plt.show()
