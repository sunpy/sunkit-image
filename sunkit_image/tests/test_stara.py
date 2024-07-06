import numpy as np

import astropy.units as u

from sunkit_image.stara import stara
from sunkit_image.tests.helpers import figure_test


def test_stara(hmi_map):
    hmi_upscaled = hmi_map.resample((512, 512) * u.pixel)
    result = stara(hmi_upscaled)
    assert isinstance(result, np.ndarray)
    assert result.shape == hmi_upscaled.data.shape
    total_true_value_count = sum(result.ravel())
    assert total_true_value_count == 5033


def test_stara_threshold_adjustment(hmi_map):
    hmi_upscaled = hmi_map.resample((512, 512) * u.pixel)
    # Apply STARA with a lower threshold, expecting to detect more features
    lower_threshold_result = stara(hmi_upscaled, threshold=2000)
    higher_threshold_result = stara(hmi_upscaled, threshold=8000)
    # A lower limb filter, would detect more features
    lower_limb_filtered_result = stara(hmi_upscaled, limb_filter=5 * u.percent)
    higher_limb_filtered_result = stara(hmi_upscaled, limb_filter=30 * u.percent)
    # Assert that the lower threshold results in more features detected
    assert lower_threshold_result.sum() > higher_threshold_result.sum(), "Lower threshold should detect more features"
    assert (
        lower_limb_filtered_result.sum() > higher_limb_filtered_result.sum()
    ), "Lower Limb filter should detect more features"


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
