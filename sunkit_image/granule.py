"""
This module contains functions that will segment images for granule detection.
"""
import logging

import numpy as np
import scipy.ndimage as sndi
import skimage
from sklearn.cluster import KMeans

import sunpy
import sunpy.map

__all__ = [
    "segment",
    "get_threshold",
    "trim_intergranules",
    "mark_faculae",
    "kmeans_segment",
    "cross_correlation",
]

METHODS = ["otsu", "li", "isodata", "mean", "minimum", "yen", "triangle"]


def segment(smap, resolution, *, skimage_method="li", mark_dim_centers=False):
    """
    Segment an optical image of the solar photosphere into tri-value maps with:

     * 0 as intergranule
     * 0.5 as faculae
     * 1 as granule

    Parameters
    ----------
    smap : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing data to segment.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    skimage_method : {"otsu", "li", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method.
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`
        Whether to mark dim granule centers as a separate category for future exploration.

    Returns
    -------
    segmented_map : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing a segmented image (with the original header).
    """
    if not isinstance(smap, sunpy.map.mapbase.GenericMap):
        raise TypeError("Input must be an instance of a sunpy.map.GenericMap")
    if skimage_method not in METHODS:
        raise TypeError("Method must be one of: " + ", ".join(METHODS))

    median_filtered = sndi.median_filter(smap.data, size=3)
    # Apply initial skimage threshold.
    threshold = get_threshold(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = trim_intergranules(segmented_image, mark=mark_dim_centers)
    # Mark faculae and get final granule and facule count.
    seg_im_markfac, faculae_count, granule_count = mark_faculae(seg_im_fixed, smap.data, resolution)
    logging.info(f"Segmentation has identified {granule_count} granules and {faculae_count} faculae")
    segmented_map = sunpy.map.Map(seg_im_markfac, smap.meta)
    return segmented_map


def get_threshold(data, method):
    """
    Get the threshold value using given skimage segmentation type.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to threshold.
    method : {"otsu", "li", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method.

    Returns
    -------
    threshold : `float`
        Threshold value.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an instance of a np.ndarray")
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
    else:
        raise ValueError("Method must be one of: " + ", ".join(METHODS))
    return threshold


def trim_intergranules(segmented_image, mark=False):
    """
    Remove the erroneous identification of intergranule material in the middle
    of granules that the pure threshold segmentation produces.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect extra intergranules.
    mark : `bool`
        If `False` (the default), remove erroneous intergranules.
        If `True`, mark them as 0.5 instead (for later examination).

    Returns
    -------
    segmented_image_fixed : `numpy.ndarray`
        The segmented image without incorrect extra intergranules.
    """
    if len(np.unique(segmented_image)) > 2:
        raise ValueError("segmented_image must only have values of 1 and 0.")
    # Float conversion for correct region labeling.
    segmented_image_fixed = np.copy(segmented_image).astype(float)
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    # Find value of the large continuous 0-valued region.
    size = 0
    for value in values:
        if len((labeled_seg[labeled_seg == value])) > size:
            real_IG_value = value
            size = len(labeled_seg[labeled_seg == value])
    # Set all other 0 regions to mark value (1 or 0.5).
    for value in values:
        if np.sum(segmented_image[labeled_seg == value]) == 0:
            if value != real_IG_value:
                if not mark:
                    segmented_image_fixed[labeled_seg == value] = 1
                elif mark:
                    segmented_image_fixed[labeled_seg == value] = 0.5
    return segmented_image_fixed


def mark_faculae(segmented_image, data, resolution):
    """
    Mark faculae separately from granules - give them a value of 2 not 1.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with faculae marked as 1.5.
    faculae_count: `int`
        The number of faculae identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of faculae.
    """
    fac_size_limit = 2  # Max size of a faculae in square arcsec.
    fac_pix_limit = fac_size_limit / resolution
    # General flux limit determined by visual inspection.
    fac_brightness_limit = np.mean(data) + 0.5 * np.std(data)
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 0.5 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    fac_count = 0
    for value in values:
        mask = np.zeros_like(segmented_image)
        mask[labeled_seg == value] = 1
        # Check that is a 1 (white) region.
        if np.sum(np.multiply(mask, segmented_image)) > 0:
            region_size = len(segmented_image_fixed[mask == 1])
            tot_flux = np.sum(data[mask == 1])
            # check that region is small.
            if region_size < fac_pix_limit:
                # Check that avg flux very high.
                if tot_flux / region_size > fac_brightness_limit:
                    segmented_image_fixed[mask == 1] = 1.5
                    fac_count += 1
    gran_count = len(values) - 1 - fac_count  # Subtract 1 for IG region.
    return segmented_image_fixed, fac_count, gran_count


def kmeans_segment(data):
    """
    Uses a k-means clustering algorithm to cluster data, in order to
    independently verify the scikit-image clustering method.

    Parameters
    ----------
    data : `numpy array`
        Data to be clustered.

    Returns
    -------
    labels : `numpy.ndarray`
        An array of labels with:

        * 0 are granules
        * 2 are intergranules
        * 1 are in-between regions
    """
    x_size = np.shape(data)[0]
    y_size = np.shape(data)[1]
    data_flat = np.reshape(data, (x_size * y_size, 1))
    labels_flat = KMeans(n_clusters=3, n_init=20).fit(data_flat).labels_
    labels = np.reshape(labels_flat, (x_size, y_size))
    # Make intergranules = 0 and granules = 1.
    group0_mean = np.mean(data[labels == 0])
    group1_mean = np.mean(data[labels == 1])
    group2_mean = np.mean(data[labels == 2])
    min_index = np.argmin([group0_mean, group1_mean, group2_mean])
    segmented_map = np.ones(labels.shape)
    segmented_map[[labels[:, :] == min_index][0]] -= 1
    return segmented_map


def cross_correlation(segment1, segment2):
    """
    Works out the cross correlation of two segmented arrays.

        -1 if the agreement between two arrays is low, 0 otherwise.

        Designed to be used with `segment` and `segment_kmeans` function.

    Parameters
    ----------
    segment1 : `numpy.ndarray`
        Main array to compare the other input array against.
    segment2 : `numpy.ndarray`
        Other array, i.e., data segmented using k-means clustering.

    Returns
    -------
    [label, confidence] : `list`
        label : `int`
            Summarizes the confidence metric:
            -1: if agreement is low (below 75%)
            0: otherwise
        confidence : `float`
            The numeric confidence metric:
            Between 0 and 1 where 0 if there is no agreement and 1 if completely agrees.
    """
    total_granules = np.count_nonzero(segment1 == 1)
    total_intergranules = np.count_nonzero(segment1 == 0)
    if total_granules == 0:
        raise ValueError("No granules in `segment1`. It is possible the clustering failed.")
    if total_intergranules == 0:
        raise ValueError("No intergranules in `segment1`. It is possible the clustering failed.")
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
    confidence = np.mean([percentage_agreement_granules, percentage_agreement_intergranules])
    if percentage_agreement_granules < 0.75 or percentage_agreement_intergranules < 0.75:
        logging.info("Low agreement with K-Means clustering. Saved output has low confidence.")
        return [-1, confidence]
    else:
        return [0, confidence]