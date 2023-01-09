import unittest
import random
import numpy as np
import sunpy
import sunpy.map as sm
from sunpy.coordinates import frames
from astropy.io import fits
import os
# import pathlib as pl
# import scipy
# import scipy.io as sio
import shutil
import astropy.units as u
from astropy.coordinates import SkyCoord
import sunkit_image.granule as granule


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ Set up for unit testing by creating toy data
        """
        #cls.ibis_testfile = 'data/IBIS/IBIS_example.sav'
        #cls.ibis_res = 0.096
        #cls.dkist_testfile = 'data/DKIST/DKIST_example.fits'
        #cls.dkist_res = 0.016
        cls.res =  0.016

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
        cls.data_map = sunpy.map.Map(mock_data, mock_header)

        # cls.dkist_fileid = 'DKIST_example'
        # cls.ibis_fileid = 'IBIS_example'
        # cls.ibis_instrument = 'IBIS'
        # cls.dkist_instrument = 'DKIST'
        cls.id = 'example'
        cls.test_method = 'li'
        cls.test_band = 'rosa_gband'
        cls.test_header = None

    @classmethod
    def tearDownClass(cls):
        """ Tear down unit testing toy data
        """

        # cls.ibis_testfile = None
        # cls.dkist_testfile = None
        cls.data_map =  None
        # cls.ibis_fileid = None
        # cls.dkist_fileid = None
        cls.id = None
        # cls.ibis_instrument = None
        # cls.dkist_instrument = None
        cls.test_method = None
        cls.test_band = None
        shutil.rmtree('test_output')

    def test_segment(self):
        """ Unit tests for segment() function
        """

        # -------- positive tests -------- :
        # data_map = funclib.sav_to_map(self.ibis_testfile, self.test_band)
        segmented = granule.segment(self.id, self.data_map,
                                    self.test_method, self.res,
                                    True, False, 'test_output/')

        test_type = type(segmented)
        # Test 1: check that the returned type is correct
        self.assertEqual(test_type, sunpy.map.mapbase.GenericMap)
        # Test 2: get an array of pixel values and check it is not empty
        initial_pix = sm.all_pixel_indices_from_map(self.data_map).value
        seg_pixels = sm.all_pixel_indices_from_map(segmented).value
        self.assertTrue(np.size(seg_pixels) > 0)
        # Test 3: check that the returned shape is unchanged
        self.assertEqual(seg_pixels.shape, initial_pix.shape)

        # -------- negative tests -------- :
        # Test 4: check that the values in the array are changed
        random.seed(42)
        # pick 10 random indices to check
        x = random.sample(list(np.arange(0, self.data_map.data.shape[0], 1)), 10)
        y = random.sample(list(np.arange(0, self.data_map.data.shape[1], 1)), 10)
        for i in range(len(x)):
            self.assertNotEqual(self.data_map.data[x[i], y[i]],
                                segmented.data[x[i], y[i]])

        # ------ error raising tests ------ :
        # Test 5: check that errors are raised for incorrect inputs
        self.assertRaises(TypeError, granule.segment, self.id,
                          np.array([[1,2,3], [1,2,3]]), self.test_method,
                          self.res)
        self.assertRaises(TypeError, granule.segment, self.id,
                          self.data_map, 'method',
                          self.res)

    def test_get_threshold(self):
        """ Unit tests for get_threshold() function
        """

        # -------- positive tests -------- :
        # Test 1: check type of output
        test_arr1 = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        threshold1 = granule.get_threshold(test_arr1, self.test_method)
        self.assertEqual(type(threshold1), np.float64)
        # Test 2: check range of output
        self.assertTrue((threshold1 > 0) * (threshold1 < np.max(test_arr1)))

        # -------- negative tests -------- :
        # Test 3: check that different arrays return different thresholds
        test_arr2 = np.array([[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]])
        threshold2 = granule.get_threshold(test_arr2, self.test_method)
        self.assertNotEqual(threshold1, threshold2)

        # ------ error raising tests ------ :
        # Test 4: check that errors are raised for incorrect inputs
        self.assertRaises(ValueError,
                          granule.get_threshold,
                          test_arr1,
                          'banana')
        self.assertRaises(ValueError,
                          granule.get_threshold,
                          [],
                          self.test_method)

    def test_trim_intergranules(self):
        """ Unit tests for trim_intergranules() function
        """

        # -------- positive tests -------- :
        # data_map = granule.sav_to_map(self.ibis_testfile, self.test_band)
        thresholded = np.uint8(self.data_map.data > np.nanmedian(self.data_map.data))
        # Test 1: check that returned array is not empty
        self.assertTrue(np.size(thresholded) > 0)
        # Test 2: check that the correct dimensions are returned
        self.assertEqual(thresholded.shape,
                         granule.trim_intergranules(thresholded).shape)
        # Test 3: mark erronous material, not remove when flag is True.
        middles_marked = \
            granule.trim_intergranules(thresholded, mark=True)
        marked_erroneous = \
            np.count_nonzero(middles_marked[middles_marked == 0.5])
        self.assertNotEqual(marked_erroneous, 0)
        # Test 4: remove when flag is False (no 0.5 values)
        middles_marked = \
            granule.trim_intergranules(thresholded, mark=False)
        marked_erroneous = \
            np.count_nonzero(middles_marked[middles_marked == 0.5])
        self.assertEqual(marked_erroneous, 0)

        # -------- negative tests -------- :
        # Test 4: check that the returned array has fewer (or same number)
        # 0-valued pixels as input array (for a data set which we
        # know by eye should have some middle sections removed)
        middles_removed = granule.trim_intergranules(thresholded)
        self.assertFalse(np.count_nonzero(middles_removed)
                         < np.count_nonzero(thresholded))

        # ------ error raising tests ------ :
        # Test 5: check that raises error if passed array is not binary
        self.assertRaises(ValueError, granule.trim_intergranules,
                          self.data_map.data)

    def test_mark_faculae(self):
        """ Unit tests for mark_faculae() function
        """

        # -------- positive tests -------- :
        #data_map = funclib.sav_to_map(self.ibis_testfile, self.test_band)
        thresholded = np.uint8(self.data_map.data > np.nanmedian(self.data_map.data))
        faculae_marked = granule.mark_faculae(thresholded,
                                              self.data_map.data,
                                              res=self.res)

        # Test 1: check that the correct dimensions are returned
        self.assertEqual(thresholded.shape,
                         faculae_marked.shape)
        # Test 2: check that returned array is not empty
        self.assertTrue(np.size(faculae_marked) > 0)

        # -------- negative tests -------- :
        # Test 3: check that the returned array has some 0.5 values (for a
        # dataset that we know has faculae by eye)
        self.assertNotEqual(len(np.where(faculae_marked == 1.5)[0]), 0)

        # ------ error raising tests ------ :
        # Test 4: check that errors are raised for incorrect inputs
        self.assertRaises(ValueError,
                          granule.mark_faculae,
                          self.data_map.data,
                          self.data_map.data,
                          res=self.res)

    def test_kmeans_segment(self):
        """ Unit tests for test_kmeans() function
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
        self.assertEqual(np.shape(clustered_array)[0], N)

        # -------- negative tests -------- :
        # Test 2: test that the returned labels don't contian a
        # label they shouldn't (should only be 0 or 1)
        non_label = 3
        count_non_label_in_cluster =\
            np.count_nonzero(clustered_array[clustered_array == non_label])
        self.assertEqual(count_non_label_in_cluster, 0)

        # ------ error raising tests ------ :
        # Test 3: should error if passed in data of wrong shape
        self.assertRaises(Exception,
                          granule.kmeans_segment,
                          array_to_be_clustered,
                          3)

    def test_cross_correlation(self):
        """ Unit tests for cross_correlation() function
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

        self.assertEqual(-1, granule.cross_correlation(test_array_1,
                                                       test_array_2)[0])

        # -------- negative tests -------- :
        # Test 1: check cross correlation isn't greater than 100% or
        # less than 0%
        self.assertFalse(granule.cross_correlation(test_array_1,
                                                   test_array_2)[1] > 1)
        self.assertFalse(granule.cross_correlation(test_array_1,
                                                   test_array_2)[1] < 0)

        # ------ error raising tests ------ :
        # Test 1: error if no granules or intergranules in skimage cluster
        test_array_1 = np.ones((test_size, test_size))
        test_array_2 = np.ones((test_size, test_size))
        self.assertRaises(Exception, granule.cross_correlation,
                          test_array_1, test_array_2)