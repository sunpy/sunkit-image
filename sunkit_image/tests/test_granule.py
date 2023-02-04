import random

import numpy as np
import pytest

import sunpy
from sunpy.map import all_pixel_indices_from_map

import sunkit_image.granule as granule


@pytest.mark.remote_data
def test_segment(test_granule_map):
    segmented = granule.segment(test_granule_map, skimage_method="li", mark_dim_centers=True)
    assert isinstance(segmented, sunpy.map.mapbase.GenericMap)
    # Check pixels are not empty.
    initial_pix = all_pixel_indices_from_map(test_granule_map).value
    seg_pixels = all_pixel_indices_from_map(segmented).value
    assert np.size(seg_pixels) > 0
    assert seg_pixels.shape == initial_pix.shape
    # Check that the values in the array are changed (pick 10 random indices to check).
    x = random.sample(list(np.arange(0, test_granule_map.data.shape[0], 1)), 10)
    y = random.sample(list(np.arange(0, test_granule_map.data.shape[1], 1)), 10)
    for i in range(len(x)):
        assert test_granule_map.data[x[i], y[i]] != segmented.data[x[i], y[i]]


@pytest.mark.remote_data
def test_segment_errors(test_granule_map):
    with pytest.raises(TypeError, match="Input must be an instance of a sunpy.map.GenericMap"):
        granule.segment(np.array([[1, 2, 3], [1, 2, 3]]))
    with pytest.raises(TypeError, match="Method must be one of: otsu, li, isodata, mean, minimum, yen, triangle"):
        granule.segment(test_granule_map, skimage_method="banana")


def test_get_threshold():
    test_arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    threshold1 = granule._get_threshold(test_arr1, "li")
    assert isinstance(threshold1, np.float64)
    # Check that different arrays return different thresholds.
    test_arr2 = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
    threshold2 = granule._get_threshold(test_arr2, "li")
    assert threshold1 != threshold2


def test_get_threshold_range():
    test_arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    threshold1 = granule._get_threshold(test_arr1, "li")
    assert 0 < threshold1 < np.max(test_arr1)


def test_get_threshold_errors():
    with pytest.raises(ValueError, match="Input data must be an instance of a np.ndarray"):
        granule._get_threshold([], "li")
    with pytest.raises(ValueError, match="Method must be one of: "):
        granule._get_threshold(np.array([[1, 2], [1, 2]]), "banana")


@pytest.mark.remote_data
def test_trim_intergranules(test_granule_map):
    thresholded = np.uint8(test_granule_map.data > np.nanmedian(test_granule_map.data))
    # Check that returned array is not empty.
    assert np.size(thresholded) > 0
    # Check that the correct dimensions are returned.
    assert thresholded.shape == granule._trim_intergranules(thresholded).shape
    # Check that erroneous material marked, not removed, when flag is True.
    middles_marked = granule._trim_intergranules(thresholded, mark=True)
    marked_erroneous = np.count_nonzero(middles_marked[middles_marked == 3])
    assert marked_erroneous != 0
    # Check that removed when flag is False (no 3 values).
    middles_marked = granule._trim_intergranules(thresholded, mark=False)
    marked_erroneous = np.count_nonzero(middles_marked[middles_marked == 3])
    assert marked_erroneous == 0
    # Check that the returned array has fewer (or same number) 0-valued pixels as input
    # array (for a data set which we know by eye should have some middle sections removed).
    middles_removed = granule._trim_intergranules(thresholded)
    assert not np.count_nonzero(middles_removed) < np.count_nonzero(thresholded)


def test_trim_intergranules_errors():
    # Check that raises error if passed array is not binary.
    data = np.random.randint(0, 10, size=(10, 10))
    with pytest.raises(ValueError, match="segmented_image must only have values of 1 and 0."):
        granule._trim_intergranules(data)


@pytest.mark.remote_data
def test_mark_brightpoint(test_granule_map):
    thresholded = np.uint8(test_granule_map.data > np.nanmedian(test_granule_map.data))
    brightpoint_marked, _, _ = granule._mark_brightpoint(thresholded, test_granule_map.data, resolution=0.016, bp_min_flux=None)
    # Check that the correct dimensions are returned.
    assert thresholded.shape == brightpoint_marked.shape
    # Check that returned array is not empty.
    assert np.size(brightpoint_marked) > 0
    # Check that the returned array has some 2 values (for a dataset that we know has brightpoints by eye).
    assert len(np.where(brightpoint_marked == 2)[0]) != 0


@pytest.mark.remote_data
def test_mark_brightpoint_error(test_granule_map):
    # Check that errors are raised for incorrect test_granule_map.
    with pytest.raises(ValueError, match="segmented_image must have only"):
        granule._mark_brightpoint(test_granule_map.data, test_granule_map.data, 0.016, bp_min_flux=None)


def test_segments_overlap_fraction(granule_minimap1):
    # Check that segments_overlap_fraction is 1 when Maps are equal.
    map = granule_minimap1
    assert granule._segments_overlap_fraction(map, map) == 1.0


def test_segments_overlap_fraction2(granule_minimap1, granule_minimap2):
    # Check that segments_overlap_fraction is between 0 and 1 when Maps are not equal. 
    map1 = granule_minimap1
    map2 = granule_minimap2
    assert granule._segments_overlap_fraction(map1 map2) <= 1
    assert not granule._segments_overlap_fraction(map1, map2) < 0


def test_segments_overlap_fraction_errors(granule_minimap3):
    # Check that error is raised if there are no granules or intergranules in image.
    map = granule_minimap3
    with pytest.raises(Exception, match="clustering failed"):
        granule._segments_overlap_fraction(map, map)
