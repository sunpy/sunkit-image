import random

import numpy as np
import pytest

import sunpy
import sunpy.map as sm

import sunkit_image.granule as granule


@pytest.fixture(scope="session")
def inputs():
    import warnings

    with warnings.catch_warnings():
        smap = sunpy.map.Map("sunkit_image/tests/granule_testdata.fits")
    test_res = 0.016
    test_method = "li"
    return smap, test_res, test_method


def test_segment1(inputs):

    smap, test_res, test_method = inputs

    segmented = granule.segment(smap, test_res, test_method, True)

    # Check returned object is SunPy map.
    assert isinstance(segmented, sunpy.map.mapbase.GenericMap)

    # Check pixels are not empty.
    initial_pix = sm.all_pixel_indices_from_map(smap).value
    seg_pixels = sm.all_pixel_indices_from_map(segmented).value
    assert np.size(seg_pixels) > 0

    # Check that the returned shape is unchanged.
    assert seg_pixels.shape == initial_pix.shape

    # Check that the values in the array are changed (pick 10 random indices to check).
    x = random.sample(list(np.arange(0, smap.data.shape[0], 1)), 10)
    y = random.sample(list(np.arange(0, smap.data.shape[1], 1)), 10)
    for i in range(len(x)):
        assert smap.data[x[i], y[i]] != segmented.data[x[i], y[i]]


def test_segment2(inputs):

    smap, test_res, test_method = inputs

    # Check that errors are raised for incorrect inputs.
    with pytest.raises(TypeError, match="Input must be sunpy map."):
        granule.segment(np.array([[1, 2, 3], [1, 2, 3]]), test_res, test_method)
    with pytest.raises(TypeError, match="Method must be one of: "):
        granule.segment(smap, test_res, "banana")


def test_get_threshold1(inputs):

    _, _, test_method = inputs

    # Check type of output.
    test_arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    threshold1 = granule.get_threshold(test_arr1, test_method)
    assert type(threshold1) is np.float64

    # Check that different arrays return different thresholds.
    test_arr2 = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
    threshold2 = granule.get_threshold(test_arr2, test_method)
    assert threshold1 != threshold2


def test_get_threshold2(inputs):

    _, _, test_method = inputs

    # Check range of output.
    test_arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    threshold1 = granule.get_threshold(test_arr1, test_method)
    assert 0 < threshold1 < np.max(test_arr1)


def test_get_threshold3(inputs):

    _, _, test_method = inputs

    # Check that errors are raised for incorrect inputs.
    with pytest.raises(ValueError, match="Input data must be an array."):
        granule.get_threshold([], test_method)
    with pytest.raises(ValueError, match="Method must be one of: "):
        granule.get_threshold(np.array([[1, 2], [1, 2]]), "banana")


def test_trim_intergranules1(inputs):

    smap, _, _ = inputs

    thresholded = np.uint8(smap.data > np.nanmedian(smap.data))

    # Check that returned array is not empty.
    assert np.size(thresholded) > 0

    # Check that the correct dimensions are returned.
    assert thresholded.shape == granule.trim_intergranules(thresholded).shape

    # Check that erronous material marked, not removed, when flag is True.
    middles_marked = granule.trim_intergranules(thresholded, mark=True)
    marked_erroneous = np.count_nonzero(middles_marked[middles_marked == 0.5])
    assert marked_erroneous != 0

    # Check that removed when flag is False (no 0.5 values).
    middles_marked = granule.trim_intergranules(thresholded, mark=False)
    marked_erroneous = np.count_nonzero(middles_marked[middles_marked == 0.5])
    assert marked_erroneous == 0


def test_trim_intergranules2(inputs):

    smap, _, _ = inputs

    thresholded = np.uint8(smap.data > np.nanmedian(smap.data))

    # Check that the returned array has fewer (or same number) 0-valued pixels as input
    # array (for a data set which we know by eye should have some middle sections removed).
    middles_removed = granule.trim_intergranules(thresholded)
    assert (np.count_nonzero(middles_removed) < np.count_nonzero(thresholded)) is False


def test_trim_intergranules3(inputs):

    smap, _, _ = inputs

    # Check that raises error if passed array is not binary.
    with pytest.raises(ValueError, match="segmented_image must have only"):
        granule.trim_intergranules(smap)


def test_mark_faculae1(inputs):

    smap, test_res, _ = inputs

    thresholded = np.uint8(smap.data > np.nanmedian(smap.data))
    faculae_marked, fac_cnt, gran_cnt = granule.mark_faculae(thresholded, smap.data, resolution=test_res)

    # Check that the correct dimensions are returned.
    assert thresholded.shape == faculae_marked.shape

    # Check that returned array is not empty.
    assert (np.size(faculae_marked) > 0) is True


def test_mark_faculae2(inputs):

    smap, test_res, _ = inputs

    thresholded = np.uint8(smap.data > np.nanmedian(smap.data))
    faculae_marked, fac_cnt, gran_cnt = granule.mark_faculae(thresholded, smap.data, resolution=test_res)

    # Check that the returned array has some 0.5 values (for a dataset that we know has
    # faculae by eye).
    assert len(np.where(faculae_marked == 1.5)[0]) != 0


def test_mark_faculae3(inputs):

    smap, test_res, _ = inputs

    # Check that errors are raised for incorrect inputs.
    with pytest.raises(
        ValueError, match="segmented_image must have only"):
        granule.mark_faculae(smap.data, smap.data, test_res)


def test_kmeans_segment1():

    N = 10
    array_to_be_clustered = np.ones((N, N))
    array_to_be_clustered[0, 0] = 1 # Fake values to cluster.
    array_to_be_clustered[0, 1] = 2
    clustered_array = granule.kmeans_segment(array_to_be_clustered)

    # Check that returns numpy array of same shape as input.
    assert np.shape(clustered_array)[0] == N


def test_kmeans_segment2():

    N = 10
    array_to_be_clustered = np.ones((N, N))
    array_to_be_clustered[0, 0] = 1 # Fake values to cluster.
    array_to_be_clustered[0, 1] = 2
    clustered_array = granule.kmeans_segment(array_to_be_clustered)

    # Check that the returned labels don't contian labels other than 0 or 1.
    non_label = 3
    count_non_label_in_cluster = np.count_nonzero(clustered_array[clustered_array == non_label])
    assert count_non_label_in_cluster == 0

    # Check that error is raised for incorrect input shape.
    with pytest.raises(Exception, match="Wrong data shape."):
        granule.kmeans_segment(array_to_be_clustered, 3)


def test_cross_correlation1():

    # Check that if arrays agree, returns 0.
    test_size = 10
    test_array_1 = np.ones((test_size, test_size))
    test_array_2 = np.ones((test_size, test_size))
    test_array_1[0, 0] = 0
    test_array_2[0, 0] = 0
    assert 0 == granule.cross_correlation(test_array_1, test_array_2)[0]


def test_cross_correlation2():

    # Check that if cross correlation is too low, returns -1.
    test_array_1 = np.ones((10, 10))
    test_array_1[0, 0] = 0
    test_array_2 = np.ones((10, 10))
    test_array_2[0, 0] = 1
    assert granule.cross_correlation(test_array_1, test_array_2)[0] == -1


def test_cross_correlation3():

    # Check that cross correlation isn't greater than 100% or less than 0%.
    assert granule.cross_correlation(test_array_1, test_array_2)[1] < 1
    assert granule.cross_correlation(test_array_1, test_array_2)[1] == 0

    # Check that error is raised if there are no granules or intergranules in image.
    test_array_1 = np.ones((10, 10))
    test_array_2 = np.ones((10, 10))
    with pytest.raises(Exception, match="clustering failed"):
        granule.cross_correlation(test_array_1, test_array_2)
