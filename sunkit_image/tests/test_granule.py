import numpy as np
import pytest

import sunpy
from sunpy.map import all_pixel_indices_from_map

from sunkit_image.granule import (
    _get_threshold,
    _mark_brightpoint,
    _trim_intergranules,
    segment,
    segments_overlap_fraction,
)

pytestmark = [pytest.mark.filterwarnings("ignore:Missing metadata for observer")]


def test_segment(granule_map):
    segmented = segment(granule_map, skimage_method="li", mark_dim_centers=True)
    assert isinstance(segmented, sunpy.map.mapbase.GenericMap)
    # Check pixels are not empty.
    initial_pix = all_pixel_indices_from_map(granule_map).value
    seg_pixels = all_pixel_indices_from_map(segmented).value
    assert np.size(seg_pixels) > 0
    assert seg_pixels.shape == initial_pix.shape
    # Check that the values in the array have changed
    assert np.any(np.not_equal(granule_map.data, segmented.data))


def test_segment_errors(granule_map):
    with pytest.raises(TypeError, match="Input must be an instance of a sunpy.map.GenericMap"):
        segment(np.array([[1, 2, 3], [1, 2, 3]]))
    with pytest.raises(ValueError, match="Method must be one of: li, otsu, yen, mean, minimum, triangle, isodata"):
        segment(granule_map, skimage_method="banana")


def test_get_threshold():
    test_arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    threshold1 = _get_threshold(test_arr1, "li")
    assert isinstance(threshold1, np.float64)
    # Check that different arrays return different thresholds.
    test_arr2 = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
    threshold2 = _get_threshold(test_arr2, "li")
    assert threshold1 != threshold2


def test_get_threshold_range():
    test_arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    threshold1 = _get_threshold(test_arr1, "li")
    assert 0 < threshold1 < np.max(test_arr1)


def test_get_threshold_errors():
    with pytest.raises(TypeError, match="Input data must be an instance of a np.ndarray"):
        _get_threshold([], "li")
    with pytest.raises(ValueError, match="Method must be one of: li, otsu, yen, mean, minimum, triangle, isodata"):
        _get_threshold(np.array([[1, 2], [1, 2]]), "banana")


def test_trim_intergranules(granule_map):
    thresholded = np.uint8(granule_map.data > np.nanmedian(granule_map.data))
    # Check that returned array is not empty.
    assert np.size(thresholded) > 0
    # Check that the correct dimensions are returned.
    assert thresholded.shape == _trim_intergranules(thresholded).shape
    # Check that erroneous zero values are caught and re-assigned
    # e.g. inside of pad region, returned array has fewer 0-valued pixels then input
    middles_removed = _trim_intergranules(thresholded)
    pad = int(np.shape(thresholded)[0] / 200)
    assert not np.count_nonzero(middles_removed[pad:-pad, pad:-pad]) < np.count_nonzero(thresholded[pad:-pad, pad:-pad])
    # Check that when mark=True, erroneous 0 values are set to 3
    middles_marked = _trim_intergranules(thresholded, mark=True)
    marked_as_3 = np.count_nonzero(middles_marked[middles_marked == 3])
    assert marked_as_3 != 0
    # Check that when mark=False, erroneous 0 values are "removed" (set to 1), returning NO 3 values
    middles_marked = _trim_intergranules(thresholded, mark=False)
    marked_as_3 = np.count_nonzero(middles_marked[middles_marked == 3])
    assert marked_as_3 == 0


def test_trim_intergranules_errors():
    rng = np.random.default_rng()
    # Check that raises error if passed array is not binary.
    data = rng.integers(low=0, high=10, size=(10, 10))
    with pytest.raises(ValueError, match="segmented_image must only have values of 1 and 0."):
        _trim_intergranules(data)


def test_mark_brightpoint(granule_map, granule_map_he):
    thresholded = np.uint8(granule_map.data > np.nanmedian(granule_map_he))
    brightpoint_marked, _, _ = _mark_brightpoint(
        thresholded,
        granule_map.data,
        granule_map_he,
        resolution=0.016,
        bp_min_flux=None,
    )
    # Check that the correct dimensions are returned.
    assert thresholded.shape == brightpoint_marked.shape
    # Check that returned array is not empty.
    assert np.size(brightpoint_marked) > 0
    # Check that the returned array has some pixels of value 2 (for a dataset that we know has brightpoints by eye).
    assert (brightpoint_marked == 2).sum() > 0


def test_mark_brightpoint_error(granule_map, granule_map_he):
    # Check that errors are raised for incorrect granule_map.
    with pytest.raises(ValueError, match="segmented_image must have only"):
        _mark_brightpoint(granule_map.data, granule_map.data, granule_map_he, resolution=0.016, bp_min_flux=None)


def test_segments_overlap_fraction(granule_minimap1):
    # Check that segments_overlap_fraction is 1 when Maps are equal.
    assert segments_overlap_fraction(granule_minimap1, granule_minimap1) == 1.0


def test_segments_overlap_fraction2(granule_minimap1, granule_minimap2):
    # Check that segments_overlap_fraction is between 0 and 1 when Maps are not equal.
    assert segments_overlap_fraction(granule_minimap1, granule_minimap2) <= 1
    assert segments_overlap_fraction(granule_minimap1, granule_minimap2) >= 0


def test_segments_overlap_fraction_errors(granule_minimap3):
    # Check that error is raised if there are no granules or intergranules in image.
    with pytest.raises(Exception, match="clustering failed"):
        segments_overlap_fraction(granule_minimap3, granule_minimap3)
