import astropy.units as u
import numpy as np

from sunkit_image.stara import stara
from sunkit_image.tests.helpers import figure_test


def test_stara(hmi_map):
    hmi_upscaled = hmi_map.resample((512, 512) * u.pixel)
    result = stara(hmi_upscaled)
    assert isinstance(result, np.ndarray)
    assert result.shape == hmi_upscaled.data.shape
    total_true_value_count = sum(result.ravel())
    assert total_true_value_count == 5033


@figure_test
def test_stara_plot(hmi_map):
    import matplotlib.pyplot as plt

    hmi_upscaled = hmi_map.resample((1024, 1024) * u.pixel)
    segmentation = stara(hmi_upscaled)
    fig = plt.figure()
    ax = plt.subplot(projection=hmi_upscaled)
    hmi_upscaled.plot(axes=ax, autoalign=True)
    ax.contour(segmentation, levels=0)
    plt.title("Sunspots identified by STARA")
    return fig
