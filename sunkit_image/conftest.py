import os
import logging
import tempfile
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import skimage

import astropy
import astropy.config.paths
from astropy.utils.data import get_pkg_data_filename

import sunpy.data.sample
import sunpy.data.test
import sunpy.map
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


@pytest.fixture(scope="session", autouse=True)
def _tmp_config_dir(request):  # NOQA: ARG001
    """
    Globally set the default config for all tests.
    """
    tmpdir = tempfile.TemporaryDirectory()

    os.environ["SUNPY_CONFIGDIR"] = str(tmpdir.name)
    astropy.config.paths.set_temp_config._temp_path = str(tmpdir.name)  # NOQA: SLF001
    astropy.config.paths.set_temp_cache._temp_path = str(tmpdir.name)  # NOQA: SLF001

    yield

    del os.environ["SUNPY_CONFIGDIR"]
    tmpdir.cleanup()
    astropy.config.paths.set_temp_config._temp_path = None  # NOQA: SLF001
    astropy.config.paths.set_temp_cache._temp_path = None  # NOQA: SLF001


@pytest.fixture()
def _undo_config_dir_patch():
    """
    Provide a way for certain tests to not have the config dir.
    """
    oridir = os.environ["SUNPY_CONFIGDIR"]
    del os.environ["SUNPY_CONFIGDIR"]
    yield
    os.environ["SUNPY_CONFIGDIR"] = oridir


@pytest.fixture(scope="session", autouse=True)
def tmp_dl_dir(request):  # NOQA: ARG001
    """
    Globally set the default download directory for the test run to a tmp dir.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["SUNPY_DOWNLOADDIR"] = tmpdir
        yield tmpdir
        del os.environ["SUNPY_DOWNLOADDIR"]


@pytest.fixture()
def _undo_download_dir_patch():
    """
    Provide a way for certain tests to not have tmp download dir.
    """
    oridir = os.environ["SUNPY_DOWNLOADDIR"]
    del os.environ["SUNPY_DOWNLOADDIR"]
    yield
    os.environ["SUNPY_DOWNLOADDIR"] = oridir


@pytest.fixture(scope="session", autouse=True)
def _hide_parfive_progress(request):  # NOQA: ARG001
    """
    Set the PARFIVE_HIDE_PROGRESS to hide the parfive progress bar in tests.
    """
    os.environ["PARFIVE_HIDE_PROGRESS"] = "True"
    yield
    del os.environ["PARFIVE_HIDE_PROGRESS"]


def pytest_runtest_teardown(item):
    # Clear the pyplot figure stack if it is not empty after the test
    # You can see these log messages by passing "-o log_cli=true" to pytest on the command line
    if HAVE_MATPLOTLIB and plt.get_fignums():
        msg = f"Removing {len(plt.get_fignums())} pyplot figure(s) " f"left open by {item.name}"
        console_logger.info(msg)
        plt.close("all")


@pytest.fixture()
def granule_map():
    return sunpy.map.Map(get_pkg_data_filename("dkist_photosphere.fits", package="sunkit_image.data.test"))


@pytest.fixture()
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


@pytest.fixture()
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


@pytest.fixture()
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


@pytest.fixture()
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


@pytest.fixture(params=["array", "map"])
def aia_171(request):
    smap = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)
    if request.param == "map":
        return smap
    return smap.data if request.param == "array" else None


@pytest.fixture()
def hmi_map():
    return sunpy.map.Map(get_test_filepath("hmi_continuum_test_lowres_data.fits"))


@pytest.fixture()
def aia171_test_clipping():
    return np.asarray([0.2, -0.3, -1.0001])


@pytest.fixture()
def aia171_test_map():
    testpath = sunpy.data.test.rootdir
    return sunpy.map.Map(Path(testpath) / "aia_171_level1.fits")


@pytest.fixture()
def aia171_test_map_layer_shape(aia171_test_map_layer):
    return aia171_test_map_layer.shape


@pytest.fixture()
def aia171_test_shift():
    return np.array([3, 5])


@pytest.fixture()
def aia171_test_template(aia171_test_shift, aia171_test_map):
    a1 = aia171_test_shift[0] + aia171_test_map.data.shape[0] // 4
    a2 = aia171_test_shift[0] + 3 * aia171_test_map.data.shape[0] // 4
    b1 = aia171_test_shift[1] + aia171_test_map.data.shape[1] // 4
    b2 = aia171_test_shift[1] + 3 * aia171_test_map.data.shape[1] // 4
    # Requires at least 32-bit floats to pass.
    return aia171_test_map.data[a1:a2, b1:b2].astype("float32")
