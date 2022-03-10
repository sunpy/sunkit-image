"""
========================
Removing Cosmic Ray Hits
========================

This example illustrates how to remove cosmic ray hits from a LASCO C2 FITS file.
using `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__.
Astroscrappy is a separate Python package and can be installed separately using ``pip`` or ``conda``.
"""

import astroscrappy
import matplotlib.pyplot as plt

from sunpy.io.file_tools import read_file
from sunpy.map import Map
from sunpy.net import Fido
from sunpy.net import attrs as a

###############################################################################
# For more details on how to download and plot LASCO FITS file see
# SunPy's example `Downloading and plotting LASCO C3 data <https://docs.sunpy.org/en/stable/generated/gallery/acquiring_data/skip_downloading_lascoC3.html>`__.
# To make this example work you need to have SunPy with all the "net" dependencies installed.


###############################################################################
# In order to download the required FITS file, we use
# `Fido <sunpy.net.fido_factory.UnifiedDownloaderFactory>`, SunPy's downloader client.
# We need to define two search variables:  a time range and the instrument.

time_range = a.Time("2000/11/09 00:06", "2000/11/09 00:07")
instrument = a.Instrument("LASCO")
detector = a.Detector("C2")
result = Fido.search(time_range, instrument)

downloaded_files = Fido.fetch(result[0])
data, header = read_file(downloaded_files[0])[0]

# Add the missing meta information to the header
header["CUNIT1"] = "arcsec"
header["CUNIT2"] = "arcsec"

###############################################################################
# With this fix we can load it into a map and plot the results.

lasco_map = Map(data, header)
fig1 = plt.figure()
lasco_map.plot()

###############################################################################
# Now we will call the `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__
# to remove the cosmic ray hits.
# This algorithm can perform well with both high and low noise levels in the original data.
# The function takes a `~numpy.ndarray` as input so we only pass the map data.
# This particular image has lots of high intensity cosmic ray hits which
# cannot be effectively removed by using the default set of parameters.
# So we reduce ``sigclip``, the Laplacian to noise ratio from 4.5 to 2 to mark more hits.
# We also reduce ``objlim``, the contrast between the Laplacian image and the fine structured image
# to clean the high intensity bright cosmic ray hits.
# We also modify the ``readnoise`` parameter to obtain better results.

mask, clean_data = astroscrappy.detect_cosmics(lasco_map.data, sigclip=2, objlim=2, readnoise=4, verbose=True)
# This returns two variables - mask is a boolean array depicting whether there is
# a cosmic ray hit at that pixel, clean_data is the cleaned image after removing those
# hits.

###############################################################################
# We can now plot the cleaned image.

clean_map1 = Map(clean_data, lasco_map.meta)

fig2 = plt.figure()
clean_map1.plot()

plt.show()
