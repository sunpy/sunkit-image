import random

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


@pytest.mark.remote_data
def test_segment(granule_map):
    segmented = segment(granule_map, skimage_method="li", mark_dim_centers=True)
    assert isinstance(segmented, sunpy.map.mapbase.GenericMap)
    # Check pixels are not empty.
    initial_pix = all_pixel_indices_from_map(granule_map).value
    seg_pixels = all_pixel_indices_from_map(segmented).value
    assert np.size(seg_pixels) > 0
    assert seg_pixels.shape == initial_pix.shape
    # Check that the values in the array are changed (pick 10 random indices to check).
    x = random.sample(list(np.arange(0, granule_map.data.shape[0], 1)), 10)
    y = random.sample(list(np.arange(0, granule_map.data.shape[1], 1)), 10)
    for i in range(len(x)):
        assert granule_map.data[x[i], y[i]] != segmented.data[x[i], y[i]]


@pytest.mark.remote_data
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
    with pytest.raises(ValueError, match="Input data must be an instance of a np.ndarray"):
        _get_threshold([], "li")
    with pytest.raises(ValueError, match="Method must be one of: li, otsu, yen, mean, minimum, triangle, isodata"):
        _get_threshold(np.array([[1, 2], [1, 2]]), "banana")


@pytest.mark.remote_data
def test_trim_intergranules(granule_map):
    thresholded = np.uint8(granule_map.data > np.nanmedian(granule_map.data))
    # Check that returned array is not empty.
    assert np.size(thresholded) > 0
    # Check that the correct dimensions are returned.
    assert thresholded.shape == _trim_intergranules(thresholded).shape
    # Check that erroneous zero values are caught and re-assigned 
    # e.g. the returned array has fewer (or same number) 0-valued pixels as input
    middles_removed = _trim_intergranules(thresholded)
    assert not np.count_nonzero(middles_removed) < np.count_nonzero(thresholded)
    # Check that when mark=True, erroneous 0 values are set to 2
    middles_marked = _trim_intergranules(thresholded, mark=True)
    marked_as_2 = np.count_nonzero(middles_marked[middles_marked == 2])
    assert marked_as_2 != 0
    # Check that when mark=False, erroneous 0 values are "removed" (set to 1), returning NO 2 values
    middles_marked = _trim_intergranules(thresholded, mark=False)
    marked_as_2 = np.count_nonzero(middles_marked[middles_marked == 2])
    assert marked_as_2 == 0


def test_trim_intergranules_errors():
    # Check that raises error if passed array is not binary.
    data = np.random.randint(0, 10, size=(10, 10))
    with pytest.raises(ValueError, match="segmented_image must only have values of 1 and 0."):
        _trim_intergranules(data)


@pytest.mark.remote_data
def test_mark_brightpoint(granule_map):
    thresholded = np.uint8(granule_map.data > np.nanmedian(granule_map.data))
    brightpoint_marked, _, _ = _mark_brightpoint(thresholded, granule_map.data, resolution=0.016, bp_min_flux=None)
    # Check that the correct dimensions are returned.
    assert thresholded.shape == brightpoint_marked.shape
    # Check that returned array is not empty.
    assert np.size(brightpoint_marked) > 0
    # Check that the returned array has some 3 values (for a dataset that we know has brightpoints by eye).
    assert (brightpoint_marked == 3).sum() == 32768


@pytest.mark.remote_data
def test_mark_brightpoint_error(granule_map):
    # Check that errors are raised for incorrect granule_map.
    with pytest.raises(ValueError, match="segmented_image must have only"):
        _mark_brightpoint(granule_map.data, granule_map.data, 0.016, bp_min_flux=None)


def test_segments_overlap_fraction(granule_minimap1):
    # Check that segments_overlap_fraction is 1 when Maps are equal.
    assert segments_overlap_fraction(granule_minimap1, granule_minimap1) == 1.0


def test_segments_overlap_fraction2(granule_minimap1, granule_minimap2):
    # Check that segments_overlap_fraction is between 0 and 1 when Maps are not equal.
    assert segments_overlap_fraction(granule_minimap1, granule_minimap2) <= 1
    assert not segments_overlap_fraction(granule_minimap1, granule_minimap2) < 0


def test_segments_overlap_fraction_errors(granule_minimap3):
    # Check that error is raised if there are no granules or intergranules in image.
    with pytest.raises(Exception, match="clustering failed"):
        segments_overlap_fraction(granule_minimap3, granule_minimap3)
