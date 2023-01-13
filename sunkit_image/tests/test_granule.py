import pytest
import random
import numpy as np
import sunpy
import sunpy.map as sm
from sunpy.coordinates import frames
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunkit_image.granule as granule

@pytest.fixture
def test_inputs():

    mock_data =  fits.open('sunkit_image/tests/granule_testdata.fits')[0].data
    coord = SkyCoord(np.nan * u.arcsec,
                np.nan * u.arcsec,
                obstime='1111-11-11 11:11',
                observer='earth',
                frame=frames.Helioprojective)
    mock_header = \
    sunpy.map.make_fitswcs_header(data=np.empty((0, 0)),
                                    coordinate=coord,
                                    reference_pixel=[np.nan, np.nan]
                                    * u.pixel,
                                    scale=[np.nan, np.nan]
                                    * u.arcsec / u.pixel,
                                    telescope='Unknown',
                                    instrument='Unknown',
                                    wavelength=np.nan * u.angstrom)
    data_map = sunpy.map.Map(mock_data, mock_header)

    test_res =  0.016
    test_id = 'example'
    test_method = 'li'
    test_band = 'rosa_gband'
    test_header = None

    return data_map, test_res, test_id, test_method, test_band, test_header

def test_segment(test_inputs):
    """ 
    Unit tests for segment() function
    """

    data_map, test_res, test_id, test_method, _, _ = test_inputs

    # -------- positive tests -------- :
    # data_map = funclib.sav_to_map(self.ibis_testfile, self.test_band)
    segmented = granule.segment(test_id, data_map,
                                test_method, test_res,
                                True, False, 'test_output/')

    # Test 1: check that the returned type is correct
    assert type(segmented) is sunpy.map.mapbase.GenericMap
    # Test 2: get an array of pixel values and check it is not empty
    initial_pix = sm.all_pixel_indices_from_map(data_map).value
    seg_pixels = sm.all_pixel_indices_from_map(segmented).value
    assert np.size(seg_pixels) > 0
    # Test 3: check that the returned shape is unchanged
    assert seg_pixels.shape == initial_pix.shape

    # -------- negative tests -------- :
    # Test 4: check that the values in the array are changed
    random.seed(42)
    # pick 10 random indices to check
    x = random.sample(list(np.arange(0, data_map.data.shape[0], 1)), 10)
    y = random.sample(list(np.arange(0, data_map.data.shape[1], 1)), 10)
    for i in range(len(x)):
        assert data_map.data[x[i], y[i]] != segmented.data[x[i], y[i]]

    # ------ error raising tests ------ :
    # Test 5: check that errors are raised for incorrect inputs
    with pytest.raises(TypeError, match = 'Input must be sunpy map.'):
        granule.segment(test_id, np.array([[1,2,3], [1,2,3]]), test_method,
                        test_res)
    with pytest.raises(TypeError, match = 'Method must be one of: '):
        granule.segment(test_id, data_map, 'banana', test_res)

def test_get_threshold(test_inputs):
    """
    Unit tests for get_threshold() function
    """

    _, _, _, test_method, _, _ = test_inputs

    # -------- positive tests -------- :
    # Test 1: check type of output
    test_arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    threshold1 = granule.get_threshold(test_arr1, test_method)
    assert type(threshold1) is np.float64
    # Test 2: check range of output
    assert ((threshold1 > 0) * (threshold1 < np.max(test_arr1))) is True

    # -------- negative tests -------- :
    # Test 3: check that different arrays return different thresholds
    test_arr2 = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
    threshold2 = granule.get_threshold(test_arr2, test_method)
    assert threshold1 != threshold2

    # ------ error raising tests ------ :
    # Test 4: check that errors are raised for incorrect inputs
    with pytest.raises(ValueError, match = 'Input data must be an array.'):
        granule.get_threshold([], test_method)
    with pytest.raises(ValueError, match = 'Method must be one of: '):
        granule.get_threshold(test_arr1, 'banana')

def test_trim_intergranules(test_inputs):
    """
    Unit tests for trim_intergranules() function
    """

    data_map, _, _, _, _, _ = test_inputs

    # -------- positive tests -------- :
    # data_map = granule.sav_to_map(self.ibis_testfile, self.test_band)
    thresholded = np.uint8(data_map.data > np.nanmedian(data_map.data))
    # Test 1: check that returned array is not empty
    assert np.size(thresholded) > 0
    # Test 2: check that the correct dimensions are returned
    assert thresholded.shape == granule.trim_intergranules(thresholded).shape
    # Test 3: mark erronous material, not remove when flag is True.
    middles_marked = \
        granule.trim_intergranules(thresholded, mark=True)
    marked_erroneous = \
        np.count_nonzero(middles_marked[middles_marked == 0.5])
    assert marked_erroneous != 0
    # Test 4: remove when flag is False (no 0.5 values)
    middles_marked = \
        granule.trim_intergranules(thresholded, mark=False)
    marked_erroneous = \
        np.count_nonzero(middles_marked[middles_marked == 0.5])
    assert marked_erroneous == 0

    # -------- negative tests -------- :
    # Test 4: check that the returned array has fewer (or same number)
    # 0-valued pixels as input array (for a data set which we
    # know by eye should have some middle sections removed)
    middles_removed = granule.trim_intergranules(thresholded)
    assert (np.count_nonzero(middles_removed) 
            < np.count_nonzero(thresholded)) is False

    # ------ error raising tests ------ :
    # Test 5: check that raises error if passed array is not binary
    with pytest.raises(ValueError,
                       match = 'segmented_image must have only values of 1 and 0'):
        granule.trim_intergranules(data_map)

def test_mark_faculae(test_inputs):
    """
    Unit tests for mark_faculae() function
    """

    data_map, test_res, _, _, _, _ = test_inputs

    # -------- positive tests -------- :
    #data_map = funclib.sav_to_map(self.ibis_testfile, self.test_band)
    thresholded = np.uint8(data_map.data > np.nanmedian(data_map.data))
    faculae_marked = granule.mark_faculae(thresholded, data_map.data,
                                          res=test_res)

    # Test 1: check that the correct dimensions are returned
    assert thresholded.shape == faculae_marked.shape
    # Test 2: check that returned array is not empty
    assert (np.size(faculae_marked) > 0) is True

    # -------- negative tests -------- :
    # Test 3: check that the returned array has some 0.5 values (for a
    # dataset that we know has faculae by eye)
    assert len(np.where(faculae_marked == 1.5)[0]) != 0

    # ------ error raising tests ------ :
    # Test 4: check that errors are raised for incorrect inputs
    with pytest.raises(ValueError, match = 'segmented_image must have only ' +
                         'values of 1, 0 an 0.5 (if dim centers marked)'):
        granule.mark_faculae(data_map.data, data_map.data, test_res)

def test_kmeans_segment():
    """
    Unit tests for test_kmeans() function
    """

    # -------- positive tests -------- :
    # Test 1: check that returns numpy array of same shape as input
    N = 10
    array_to_be_clustered = np.ones((N, N))
    # give some fake different values, so kmeans has something to cluster,
    array_to_be_clustered[0, 0] = 1
    array_to_be_clustered[0, 1] = 2
    clustered_array = granule.kmeans_segment(array_to_be_clustered,
                                                llambda_axis=-1)
    assert np.shape(clustered_array)[0] == N

    # -------- negative tests -------- :
    # Test 2: test that the returned labels don't contian a
    # label they shouldn't (should only be 0 or 1)
    non_label = 3
    count_non_label_in_cluster =\
        np.count_nonzero(clustered_array[clustered_array == non_label])
    assert count_non_label_in_cluster == 0

    # ------ error raising tests ------ :
    # Test 3: should error if passed in data of wrong shape
    with pytest.raises(Exception, match = 'Wrong data shape.'):
        granule.kmeans_segment(array_to_be_clustered, 3)

def test_cross_correlation(self):
    """
    Unit tests for cross_correlation() function
    """

    # -------- positive tests -------- :
    # Test 1: if arrays agree, return 0:
    test_size = 10
    test_array_1 = np.ones((test_size, test_size))
    test_array_2 = np.ones((test_size, test_size))
    test_array_1[0, 0] = 0
    test_array_2[0, 0] = 0
    self.assertEqual(0, granule.cross_correlation(test_array_1,
                                                    test_array_2)[0])

    # Test 2: if cross correlation too low, return -1:
    test_array_1 = np.ones((test_size, test_size))
    test_array_1[0, 0] = 0
    test_array_2 = np.zeros((test_size, test_size))
    test_array_2[0, 0] = 1
    assert granule.cross_correlation(test_array_1, test_array_2)[0] == -1

    # -------- negative tests -------- :
    # Test 1: check cross correlation isn't greater than 100% or
    # less than 0%
    assert granule.cross_correlation(test_array_1, test_array_2)[1] < 1
    assert granule.cross_correlation(test_array_1, test_array_2)[1] > 0

    # ------ error raising tests ------ :
    # Test 1: error if no granules or intergranules in skimage cluster
    test_array_1 = np.ones((test_size, test_size))
    test_array_2 = np.ones((test_size, test_size))
    with pytest.raises(Exception, match = 'clustering problematic'):
        granule.cross_correlation(test_array_1, test_array_2)