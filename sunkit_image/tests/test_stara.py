import numpy as np

from sunkit_image.stara import stara
from sunkit_image.tests.helpers import figure_test


def test_stara(hmi_map):
    result = stara(hmi_map)
    assert isinstance(result, np.ndarray)
    assert result.shape == hmi_map.data.shape
    assert not np.any(result)


@figure_test
def test_stara_plot(hmi_map):
    import matplotlib.pyplot as plt

    segmentation = stara(hmi_map)
    fig = plt.figure()
    ax = plt.subplot(projection=hmi_map)
    ax.contour(segmentation, levels=[0.5])
    return fig
