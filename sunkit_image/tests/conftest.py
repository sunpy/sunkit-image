import importlib.util
import os
import tempfile

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

# Force MPL to use non-gui backends for testing.
try:
    import matplotlib as mpl
except ImportError:
    pass
else:
    mpl.use("Agg")

# Don't actually import pytest_remotedata because that can do things to the
# entrypoints code in pytest.
remotedata_spec = importlib.util.find_spec("pytest_remotedata")
HAVE_REMOTEDATA = remotedata_spec is not None

# Do not collect the sample data file because this would download the sample data.
collect_ignore = ["data/sample.py"]


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


def pytest_runtest_setup(item):
    """
    Pytest hook to skip all tests that have the mark 'remotedata' if the
    pytest_remotedata plugin is not installed.
    """
    if isinstance(item, pytest.Function) and "remote_data" in item.keywords and not HAVE_REMOTEDATA:
        pytest.skip("skipping remotedata tests as pytest-remotedata is not installed")


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
def mock_hmi_map():
    data = np.random.default_rng(42).random((100, 100))
    header = {
        "date-obs": "2022-01-01T00:00:00.000",
        "crpix1": 50,
        "crpix2": 50,
        "cdelt1": 1,
        "cdelt2": 1,
        "crval1": 0,
        "crval2": 0,
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "rsun_obs": 1000,
    }
    return sunpy.map.GenericMap(data, header)
