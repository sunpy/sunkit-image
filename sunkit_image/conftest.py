import os
import tempfile
import importlib

import numpy as np
import pytest

import astropy
import astropy.config.paths
import sunpy.map
from sunpy.coordinates import Helioprojective, get_earth
from sunpy.data import manager
from sunpy.map.header_helper import make_fitswcs_header

# Force MPL to use non-gui backends for testing.
try:
    import matplotlib
except ImportError:
    pass
else:
    matplotlib.use("Agg")

# Don't actually import pytest_remotedata because that can do things to the
# entrypoints code in pytest.
remotedata_spec = importlib.util.find_spec("pytest_remotedata")
HAVE_REMOTEDATA = remotedata_spec is not None

# Do not collect the sample data file because this would download the sample data.
collect_ignore = ["data/sample.py"]


@pytest.fixture(scope="session", autouse=True)
def tmp_config_dir(request):
    """
    Globally set the default config for all tests.
    """
    tmpdir = tempfile.TemporaryDirectory()

    os.environ["SUNPY_CONFIGDIR"] = str(tmpdir.name)
    astropy.config.paths.set_temp_config._temp_path = str(tmpdir.name)
    astropy.config.paths.set_temp_cache._temp_path = str(tmpdir.name)

    yield

    del os.environ["SUNPY_CONFIGDIR"]
    tmpdir.cleanup()
    astropy.config.paths.set_temp_config._temp_path = None
    astropy.config.paths.set_temp_cache._temp_path = None


@pytest.fixture()
def undo_config_dir_patch():
    """
    Provide a way for certain tests to not have the config dir.
    """
    oridir = os.environ["SUNPY_CONFIGDIR"]
    del os.environ["SUNPY_CONFIGDIR"]
    yield
    os.environ["SUNPY_CONFIGDIR"] = oridir


@pytest.fixture(scope="session", autouse=True)
def tmp_dl_dir(request):
    """
    Globally set the default download directory for the test run to a tmp dir.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["SUNPY_DOWNLOADDIR"] = tmpdir
        yield tmpdir
        del os.environ["SUNPY_DOWNLOADDIR"]


@pytest.fixture()
def undo_download_dir_patch():
    """
    Provide a way for certain tests to not have tmp download dir.
    """
    oridir = os.environ["SUNPY_DOWNLOADDIR"]
    del os.environ["SUNPY_DOWNLOADDIR"]
    yield
    os.environ["SUNPY_DOWNLOADDIR"] = oridir


def pytest_runtest_setup(item):
    """
    pytest hook to skip all tests that have the mark 'remotedata' if the
    pytest_remotedata plugin is not installed.
    """
    if isinstance(item, pytest.Function):
        if "remote_data" in item.keywords and not HAVE_REMOTEDATA:
            pytest.skip("skipping remotedata tests as pytest-remotedata is not installed")


@pytest.fixture(scope="session")
@pytest.mark.remote_data
@manager.require(
    "granule_fits",
    "https://github.com/sunpy/data/raw/main/sunkit-image/granule_testdata.fits",
    "a118b15466dcce88e140e3235a787d99eb564f038761cb166b5f41a94b945ba9",
)
def granule_map():
    return sunpy.map.Map(manager.get("granule_fits"))


@pytest.fixture
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
    map = sunpy.map.GenericMap(arr, header)
    return map


@pytest.fixture
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
    map = sunpy.map.GenericMap(arr, header)
    return map


@pytest.fixture
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
    map = sunpy.map.GenericMap(arr, header)
    return map
