import importlib.util
import logging

import astropy
import astropy.config.paths
import numpy as np
import pytest
import skimage
import sunpy.data.sample
import sunpy.map
from astropy.utils.data import get_pkg_data_filename
from sunpy.coordinates import Helioprojective, get_earth
from sunpy.map.header_helper import make_fitswcs_header

from sunkit_image.data.test import get_test_filepath

# Force MPL to use non-gui backends for testing.
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
    mpl.use("Agg")
except ImportError:
    HAVE_MATPLOTLIB = False

# Don't actually import pytest_remotedata because that can do things to the
# entrypoints code in pytest.
remotedata_spec = importlib.util.find_spec("pytest_remotedata")
HAVE_REMOTEDATA = remotedata_spec is not None
# Do not collect the sample data file because this would download the sample data.
collect_ignore = ["data/sample.py"]
console_logger = logging.getLogger()
console_logger.setLevel("INFO")


def pytest_runtest_teardown(item):
    # Clear the pyplot figure stack if it is not empty after the test
    # You can see these log messages by passing "-o log_cli=true" to pytest on the command line
    if HAVE_MATPLOTLIB and plt.get_fignums():
        msg = f"Removing {len(plt.get_fignums())} pyplot figure(s) " f"left open by {item.name}"
        console_logger.info(msg)
        plt.close("all")


@pytest.fixture(scope="session")
def granule_map():
    return sunpy.map.Map(get_pkg_data_filename("dkist_photosphere.fits", package="sunkit_image.data.test"))


@pytest.fixture(scope="session")
def granule_map_he():
    granule_map = sunpy.map.Map(get_pkg_data_filename("dkist_photosphere.fits", package="sunkit_image.data.test"))
    # min-max normalization to [0, 1]
    map_norm = (granule_map.data - np.nanmin(granule_map.data)) / (
        np.nanmax(granule_map.data) - np.nanmin(granule_map.data)
    )
    return skimage.filters.rank.equalize(
        skimage.util.img_as_ubyte(map_norm),
        footprint=skimage.morphology.disk(radius=100),
    )


@pytest.fixture(scope="session")
def granule_minimap1():
    # Array with "intergranule region"
    arr = np.ones((10, 10))
    arr[0, 0] = 0
    observer = get_earth()
    frame = Helioprojective(observer=observer, obstime=observer.obstime)
    ref_coord = astropy.coordinates.SkyCoord(0, 0, unit="arcsec", frame=frame)
    header = make_fitswcs_header(
        arr,
        ref_coord,
    )
    return sunpy.map.GenericMap(arr, header)


@pytest.fixture(scope="session")
def granule_minimap2():
    # Modified array with "intergranule region"
    arr = np.ones((10, 10))
    arr[1, 1] = 0
    observer = get_earth()
    frame = Helioprojective(observer=observer, obstime=observer.obstime)
    ref_coord = astropy.coordinates.SkyCoord(0, 0, unit="arcsec", frame=frame)
    header = make_fitswcs_header(
        arr,
        ref_coord,
    )
    return sunpy.map.GenericMap(arr, header)


@pytest.fixture(scope="session")
def granule_minimap3():
    # Array with no "intergranule region"
    arr = np.ones((10, 10))
    observer = get_earth()
    frame = Helioprojective(observer=observer, obstime=observer.obstime)
    ref_coord = astropy.coordinates.SkyCoord(0, 0, unit="arcsec", frame=frame)
    header = make_fitswcs_header(
        arr,
        ref_coord,
    )
    return sunpy.map.GenericMap(arr, header)


@pytest.fixture(scope="session", params=["array", "map"])
def aia_171(request):
    smap = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
    if request.param == "map":
        return smap
    return smap.data if request.param == "array" else None


@pytest.fixture(scope="session")
def hmi_map():
    hmi_file = get_test_filepath("hmi_continuum_test_lowres_data.fits")
    return sunpy.map.Map(hmi_file)
