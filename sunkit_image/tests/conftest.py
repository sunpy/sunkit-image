import importlib.util
import logging
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
    # Creating the data array. The current data is extracted from a heavily downsampled hmi map
    data = np.array(
        [
            [
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                280.25,
                331.25,
                359.0,
                263.75,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
            ],
            [
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                2351.75,
                29463.5,
                33449.75,
                34484.75,
                34946.75,
                33201.5,
                29416.25,
                502.25,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
            ],
            [
                -1000.0,
                -1000.0,
                -1000.0,
                27950.0,
                34262.0,
                37859.75,
                39495.5,
                39088.25,
                40903.25,
                41133.5,
                38771.0,
                34950.5,
                26601.5,
                -1000.0,
                -1000.0,
                -1000.0,
            ],
            [
                -1000.0,
                -1000.0,
                28602.5,
                36143.75,
                40108.25,
                40595.0,
                42338.75,
                42429.5,
                41880.5,
                43013.75,
                39905.75,
                38692.25,
                35054.0,
                27640.25,
                -1000.0,
                -1000.0,
            ],
            [
                -1000.0,
                18795.5,
                34970.75,
                39134.0,
                41387.0,
                42638.75,
                44450.0,
                46085.0,
                47317.25,
                45444.5,
                42513.5,
                43598.75,
                41090.0,
                33509.75,
                405.5,
                -1000.0,
            ],
            [
                -1000.0,
                30634.25,
                38288.75,
                38823.5,
                44641.25,
                45539.75,
                47541.5,
                43559.0,
                48638.75,
                43122.5,
                41951.0,
                42267.5,
                41456.75,
                36792.5,
                29782.25,
                -1000.0,
            ],
            [
                258.5,
                33442.25,
                39335.75,
                44685.5,
                47300.75,
                44837.75,
                47569.25,
                44345.75,
                46291.25,
                45128.0,
                44091.5,
                47301.5,
                40665.5,
                39746.75,
                33491.0,
                194.0,
            ],
            [
                572.0,
                35393.0,
                40973.75,
                43319.75,
                42997.25,
                47800.25,
                46361.75,
                49904.0,
                48908.0,
                47563.25,
                48032.0,
                44780.75,
                44612.0,
                40472.0,
                34438.25,
                278.75,
            ],
            [
                530.0,
                34798.25,
                40607.0,
                45254.75,
                46691.75,
                45933.5,
                47216.0,
                45968.75,
                47096.0,
                47054.0,
                50135.75,
                43784.75,
                41009.75,
                38718.5,
                35653.25,
                268.25,
            ],
            [
                233.0,
                34026.5,
                38291.75,
                43835.0,
                43817.0,
                47188.25,
                46068.5,
                49135.25,
                45947.0,
                46414.25,
                44297.75,
                43886.75,
                40540.25,
                39104.75,
                32025.5,
                185.75,
            ],
            [
                -1000.0,
                31061.75,
                38462.0,
                40897.25,
                45355.25,
                13506.5,
                45377.75,
                46795.25,
                46757.75,
                44039.0,
                43771.25,
                42190.25,
                40165.25,
                36429.5,
                29181.5,
                -1000.0,
            ],
            [
                -1000.0,
                20867.75,
                36146.75,
                41623.25,
                42009.5,
                43008.5,
                44100.5,
                46106.75,
                43798.25,
                41219.75,
                42207.5,
                41275.25,
                39542.0,
                34676.75,
                475.25,
                -1000.0,
            ],
            [
                -1000.0,
                -1000.0,
                28699.25,
                36770.75,
                38501.0,
                43415.75,
                41420.75,
                43828.25,
                42236.0,
                43340.0,
                40098.5,
                38181.5,
                36518.0,
                27592.25,
                -1000.0,
                -1000.0,
            ],
            [
                -1000.0,
                -1000.0,
                -1000.0,
                28475.0,
                33800.75,
                36760.25,
                40776.5,
                39202.25,
                39293.75,
                38585.0,
                36566.0,
                34690.25,
                27785.75,
                -1000.0,
                -1000.0,
                -1000.0,
            ],
            [
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                19496.75,
                30044.0,
                33616.25,
                34552.25,
                33396.5,
                33146.75,
                30135.5,
                983.75,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
            ],
            [
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                261.5,
                393.5,
                383.75,
                262.25,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
                -1000.0,
            ],
        ]
    )
    # Create the header with appropriate metadata
    header = {
        "simple": "True",
        "bitpix": -64,
        "naxis": 2,
        "naxis2": 16,
        "naxis1": 16,
        "date": "2024-05-12T09:54:44",
        "date-obs": "2024-05-08T16:54:35.10",
        "wavelnth": 6173.0,
        "bunit": "DN/s",
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "crpix1": 8.45965528515625,
        "crpix2": 8.5173759453125,
        "crval1": 0,
        "crval2": 0,
        "cdelt1": 129.034752,
        "cdelt2": 129.034752,
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "crota2": 179.929596,
        "rsun_obs": 950.800171,
        "rsun_ref": 696000000,
        "wcsname": "Helioprojective-cartesian",
        "dsun_obs": 150989471585.0,
        "crln_obs": 330.716064,
        "crlt_obs": -3.348247,
        "car_rot": 2284,
    }

    return sunpy.map.GenericMap(data, header)
