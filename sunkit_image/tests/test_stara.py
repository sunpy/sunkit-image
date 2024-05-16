import numpy as np

from sunkit_image.stara import stara
from sunkit_image.tests.helpers import figure_test


def test_stara(mock_hmi_map):
    result = stara(mock_hmi_map)
    assert isinstance(result, np.ndarray)
    assert result.shape == mock_hmi_map.data.shape


@figure_test
def test_stara_plot(mock_hmi_map):
    import matplotlib.pyplot as plt

    segmentation = stara(mock_hmi_map)
    fig = plt.figure()
    ax = plt.subplot(projection=mock_hmi_map)
    ax.contour(segmentation, levels=[0.5])
    return fig
