import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.time import Time
from sunpy.map import GenericMap

from sunkit_image.stara import get_regions, stara
from sunkit_image.tests.helpers import figure_test


@pytest.fixture()
def mock_map():
    np.random.seed(42)
    data = np.random.rand(100, 100)
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
    return GenericMap(data, header)


def test_stara(mock_map):
    result = stara(mock_map)
    assert isinstance(result, np.ndarray)
    assert result.shape == mock_map.data.shape


def test_get_regions(mock_map):
    segmentation = np.zeros((100, 100))
    segmentation[40:60, 40:60] = 1
    print(segmentation)
    result = get_regions(segmentation, mock_map)
    print(result)
    assert isinstance(result, QTable)
    assert len(result) == 1
    assert result["label"][0] == 1
    assert 40 < result["centroid-0"][0] < 60
    assert 40 < result["centroid-1"][0] < 60
    assert result["area"][0] == 400
    assert result["obstime"][0] == Time("2022-01-01T00:00:00.000")
    assert isinstance(result["center_coord"][0], SkyCoord)
    assert result["center_coord"][0].frame.name == "heliographic_stonyhurst"


@figure_test
def test_stara_plot(mock_map):
    import matplotlib.pyplot as plt

    segmentation = stara(mock_map)
    plt.figure()
    ax = plt.subplot(projection=mock_map)
    mock_map.plot()
    ax.contour(segmentation, levels=[0.5])
    plt.show()