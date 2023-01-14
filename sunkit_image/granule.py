"""
This module contains functions that will segment images for granule detection.
"""

import numpy as np
import scipy.ndimage as sndi
import skimage
from sklearn.cluster import KMeans as KMeans

import sunpy
import sunpy.map

__all__ = [
    "segment",
    "get_threshold",
    "trim_intergranules",
    "mark_faculae",
    "segment_kmeans",
    "cross_correlation",
]


def segment(data_map, skimage_method, res, mark_dim_centers=False):
    """
    Segment optical image of the solar photosphere into tri-value maps
    with 0 = intergranule, 0.5 = faculae, 1 = granule.

    Parameters
    ----------
    data_map : `SunPy map`
        SunPy map containing data to segment
    skimage_method : `str`
        skimage thresholding method, with options 'otsu', 'li', 'isodata',
        'mean', 'minimum', 'yen', and 'triangle'.
        Note - depending on input data, one or more of these methods may be
        signifcantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' overidentifies granules.
    res : `float`
        Spatial resolution (arcsec/pixel) of the data
    mark_dim_centers : `bool`
        Whether to mark dim granule centers as a seperate catagory for future
        exploration.

    Returns
    -------
    segmented_map : `SunPy map`
        SunPy map containing segmentated image (with the original header)
    """

    if not isinstance(data_map, sunpy.map.mapbase.GenericMap):
        raise TypeError("Input must be sunpy map.")

    methods = ["otsu", "li", "isodata", "mean", "minimum", "yen", "triangle"]
    if skimage_method not in methods:
        raise TypeError("Method must be one of: " + str(methods))

    data = data_map.data
    header = data_map.meta

    # apply median filter
    median_filtered = sndi.median_filter(data, size=3)

    # apply threshold
    threshold = get_threshold(median_filtered, skimage_method)

    # initial skimage segmentation
    segmented_image = np.uint8(median_filtered > threshold)

    # fix the extra IGM bits in the middle of granules
    seg_im_fixed = trim_intergranules(segmented_image, mark=mark_dim_centers)

    # mark faculae and get final granule and facule count
    seg_im_markfac, fac_cnt, gran_cnt = mark_faculae(seg_im_fixed, data, res)
    print("Segmentation has identified " + str(gran_cnt) + " granules and " + str(fac_cnt) + "faculae")

    # convert segmentated image back into SunPy map with original header
    segmented_map = sunpy.map.Map(seg_im_markfac, header)

    return segmented_map


def get_threshold(data, method):
    """
    Get the threshold value using given skimage segmentation type.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to threshold.
    method : `str`
        Skimage thresholding method - options are 'otsu', 'li', 'isodata',
        'mean', 'minimum', 'yen', 'triangle'.

    Returns
    -------
    threshold `float`:
        Threshold value.
    """

    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an array.")

    methods = ["otsu", "li", "isodata", "mean", "minimum", "yen", "triangle"]
    if method not in methods:
        raise ValueError("Method must be one of: " + str(methods))
    if method == "otsu":
        threshold = skimage.filters.threshold_otsu(data)
    elif method == "li":
        threshold = skimage.filters.threshold_li(data)
    elif method == "yen":
        threshold = skimage.filters.threshold_yen(data)
    elif method == "mean":
        threshold = skimage.filters.threshold_mean(data)
    elif method == "minimum":
        threshold = skimage.filters.threshold_minimum(data)
    elif method == "triangle":
        threshold = skimage.filters.threshold_triangle(data)
    elif method == "isodata":
        threshold = skimage.filters.threshold_isodata(data)

    return threshold


def trim_intergranules(segmented_image, mark=False):
    """
    Remove the erronous idenfication of intergranule material in the middle of
    granules that pure threshold segmentation produces.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect extra intergranules.
    mark : `bool`
        If False, remove erronous intergranules. If True, mark them as 0.5
        instead (for later examination).


    Returns
    -------
    segmented_image_fixed : `numpy.ndarray`
        The segmented image without incorrect extra intergranules.
    """

    if len(np.unique(segmented_image)) > 2:
        raise ValueError("segmented_image must have only values of 1 and 0")

    segmented_image_fixed = np.copy(segmented_image).astype(float)
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    # find value of the large continuous 0-valued region
    size = 0
    for value in values:
        if len((labeled_seg[labeled_seg == value])) > size:
            real_IG_value = value
            size = len(labeled_seg[labeled_seg == value])

    # set all other 0 regions to mark value (1 or 0.5)
    for value in values:
        if np.sum(segmented_image[labeled_seg == value]) == 0:
            if value != real_IG_value:
                if not mark:
                    segmented_image_fixed[labeled_seg == value] = 1
                elif mark:
                    segmented_image_fixed[labeled_seg == value] = 0.5

    return segmented_image_fixed


def mark_faculae(segmented_image, data, res):
    """
    Mark faculae seperatley from granules - give them a value of 2 not 1.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original flux values.
    res : `float`
        Spatial resolution (arcsec/pixel) of the data.

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with faculae marked as 1.5.
    """

    fac_size_limit = 2  # max size of a faculae in sqaure arcsec
    fac_pix_limit = fac_size_limit / res
    fac_brightness_limit = np.mean(data) + 0.5 * np.std(data)

    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0, " + "an 0.5 (if dim centers marked)")

    segmented_image = segmented_image
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    fac_count = 0
    for value in values:
        mask = np.zeros_like(segmented_image)
        mask[labeled_seg == value] = 1
        # check that is a 1 (white) region
        if np.sum(np.multiply(mask, segmented_image)) > 0:
            region_size = len(segmented_image_fixed[mask == 1])
            tot_flux = np.sum(data[mask == 1])
            # check that region is small
            if region_size < fac_pix_limit:
                # check that avg flux very high
                if tot_flux / region_size > fac_brightness_limit:
                    segmented_image_fixed[mask == 1] = 1.5
                    fac_count += 1

    gran_count = len(values) - 1 - fac_count  # subtract 1 for IG region

    return segmented_image_fixed, fac_count, gran_count


def kmeans_segment(data, llambda_axis=-1):
    """
    kmeans clustering: uses a kmeans algorithm to cluster data, in order to
    independently verify the skimage clustering method (e.g using
    cross_correlation() below).

    Parameters
    ----------
    data : `numpy array`
        Data to be clustered.
    llambda_axis : `int`
        Index for wavelength, -1 if scalar array.

    Returns
    -------
    labels : `numpy.ndarray`
        An array of labels, with 0 = granules, 2 = intergranules,
        1 = in-between.
    """

    if llambda_axis not in [-1, 2]:
        raise Exception(
            "Wrong data shape. \
        (either scalar or (x, y, llambda) )"
        )
    n_clusters = 3
    n_init = 20
    x_size = np.shape(data)[0]
    y_size = np.shape(data)[1]
    if llambda_axis == -1:
        data_flat = np.reshape(data, (x_size * y_size, 1))
        labels_flat = KMeans(n_clusters).fit(data_flat).labels_
        labels = np.reshape(labels_flat, (x_size, y_size))
    else:
        llambda_size = np.shape(data)[llambda_axis]
        data = np.reshape(data, (x_size * y_size, llambda_size))
        labels = np.reshape(Kmeans(n_clusters, n_init).fit(data), (x_size, y_size))

    # make intergranules 0, granules 1:
    group0_mean = np.mean(data[labels == 0])
    group1_mean = np.mean(data[labels == 1])
    group2_mean = np.mean(data[labels == 2])

    # intergranules
    min_index = np.argmin([group0_mean, group1_mean, group2_mean])
    segmented_map = np.ones(labels.shape)

    segmented_map[[labels[:, :] == min_index][0]] -= 1

    return segmented_map


def cross_correlation(segment1, segment2):
    """
    Return -1 and print a message if the agreement between two arrays is low, 0
    otherwise. Designed to be used with segment and segment_kmeans function.

    Parameters
    ----------
    segment1 : `numpy.ndarray`
        'main' array to compare the other input array against.
    segment2 : `numpy.ndarray`
        'other' array (i.e. data segmented using kmeans).

    Returns
    -------
    [label, confidence] : `list`
        'Label' (int) summarizes the confidence metric -
            -1: if agreement is low (below 75%)
            0: otherwise
        'confidence' (float) is the numeric confidence metric -
            float between 0 and 1 (0 if no agreement, 1 if completely agrees).
    """

    total_granules = np.count_nonzero(segment1 == 1)
    total_intergranules = np.count_nonzero(segment1 == 0)

    if total_granules == 0:
        raise Exception("clustering problematic (no granules found)")

    if total_intergranules == 0:
        raise Exception("clustering problematic (no intergranules found)")

    x_size = np.shape(segment1)[0]
    y_size = np.shape(segment1)[1]

    granule_agreement_count = 0
    intergranule_agreement_count = 0
    for i in range(x_size):
        for j in range(y_size):
            if segment1[i, j] == 1 and segment2[i, j] == 1:
                granule_agreement_count += 1
            elif segment1[i, j] == 0 and segment2[i, j] == 0:
                intergranule_agreement_count += 1

    percentage_agreement_granules = granule_agreement_count / total_granules
    percentage_agreement_intergranules = intergranule_agreement_count / total_intergranules
    try:
        confidence = np.mean([percentage_agreement_granules, percentage_agreement_intergranules])
    except TypeError:
        confidence = 0

    if percentage_agreement_granules < 0.75 or percentage_agreement_intergranules < 0.75:
        print(
            "Low agreement with K-Means clustering. \
                         Saved output has low confidence."
        )
        return [-1, confidence]
    else:
        return [0, confidence]
