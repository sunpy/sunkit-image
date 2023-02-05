"""
This module contains functions that will segment images for granule detection.
"""
import logging

import matplotlib
import numpy as np
import scipy
import skimage

import sunpy
import sunpy.map

__all__ = ["segment", "segments_overlap_fraction"]


def segment(smap, *, skimage_method="li", mark_dim_centers=False, bp_min_flux=None):
    """
    Segment an optical image of the solar photosphere into tri-value maps with:

     * 0 as intergranule
     * 1 as granule
     * 2 as brightpoint

    If mark_dim_centers is set to True, an additional label, 3, will be assigned to
    dim grnanule centers.

    Parameters
    ----------
    smap : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing data to segment. Must have square pixels.
    skimage_method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}, optional
        scikit-image thresholding method, defaults to "li".
        Depending on input data, one or more of these methods may be
        significantly better or worse than the others. Typically, 'li', 'otsu',
        'mean', and 'isodata' are good choices, 'yen' and 'triangle' over-
        identify intergranule material, and 'minimum' over identifies granules.
    mark_dim_centers : `bool`, optional
        Whether to mark dim granule centers as a separate category for future exploration.
    bp_min_flux : `float`, optional
        Minimum flux per pixel for a region to be considered a brightpoint.
        Default is `None` which will use data mean + 0.5 * sigma.

    Returns
    -------
    segmented_map : `~sunpy.map.GenericMap`
        `~sunpy.map.GenericMap` containing a segmented image (with the original header).
    """
    if not isinstance(smap, sunpy.map.mapbase.GenericMap):
        raise TypeError("Input must be an instance of a sunpy.map.GenericMap")
    if smap.scale[0].value == smap.scale[1].value:
        resolution = smap.scale[0].value
    else:
        raise ValueError("Currently only maps with square pixels are supported.")
    median_filtered = scipy.ndimage.median_filter(smap.data, size=3)
    # Apply initial skimage threshold.
    threshold = _get_threshold(median_filtered, skimage_method)
    segmented_image = np.uint8(median_filtered > threshold)
    # Fix the extra intergranule material bits in the middle of granules.
    seg_im_fixed = _trim_intergranules(segmented_image, mark=mark_dim_centers)
    # Mark brightpoint and get final granule and brightpoint count.
    seg_im_markbp, brightpoint_count, granule_count = _mark_brightpoint(
        seg_im_fixed, smap.data, resolution, bp_min_flux
    )
    logging.info(f"Segmentation has identified {granule_count} granules and {brightpoint_count} brightpoint")
    # Create output map using input wcs and adding colormap such that 0 (intergranules) = black, 1 (granule) = white, 2 (brightpoints) = yellow, 3 (dim_centers) = blue.
    segmented_map = sunpy.map.Map(seg_im_markbp, smap.wcs)
    cmap = matplotlib.colors.ListedColormap(["black", "white", "#ffc406", "blue"])
    norm = matplotlib.colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=cmap.N)
    segmented_map.plot_settings["cmap"] = cmap
    segmented_map.plot_settings["norm"] = norm
    return segmented_map


def _get_threshold(data, method):
    """
    Get the threshold value using given skimage segmentation type.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data to threshold.
    method : {"li", "otsu", "isodata", "mean", "minimum", "yen", "triangle"}
        scikit-image thresholding method.

    Returns
    -------
    threshold : `float`
        Threshold value.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be an instance of a np.ndarray")
    method = method.lower()
    method_funcs = {
        "li": skimage.filters.threshold_li,
        "otsu": skimage.filters.threshold_otsu,
        "yen": skimage.filters.threshold_yen,
        "mean": skimage.filters.threshold_mean,
        "minimum": skimage.filters.threshold_minimum,
        "triangle": skimage.filters.threshold_triangle,
        "isodata": skimage.filters.threshold_isodata,
    }
    if method not in method_funcs:
        raise ValueError("Method must be one of: " + ", ".join(list(method_funcs.keys())))
    threshold = method_funcs[method](data)
    return threshold


def _trim_intergranules(segmented_image, mark=False):
    """
    Remove the erroneous identification of intergranule material in the middle
    of granules that the pure threshold segmentation produces.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect extra intergranules.
    mark : `bool`
        If `False` (the default), remove erroneous intergranules.
        If `True`, mark them as 2 instead (for later examination).

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
    # Set all other 0 regions to mark value (2).
    for value in values:
        if np.sum(segmented_image[labeled_seg == value]) == 0:
            if value != real_IG_value:
                if not mark:
                    segmented_image_fixed[labeled_seg == value] = 1
                elif mark:
                    segmented_image_fixed[labeled_seg == value] = 2
    return segmented_image_fixed


def _mark_brightpoint(segmented_image, data, resolution, bp_min_flux=None):
    """
    Mark brightpoints separately from granules - give them a value of 3.

    Parameters
    ----------
    segmented_image : `numpy.ndarray`
        The segmented image containing incorrect middles.
    data : `numpy array`
        The original image.
    resolution : `float`
        Spatial resolution (arcsec/pixel) of the data.
    bp_min_flux : `float`, optional
        Minimum flux per pixel for a region to be considered a brightpoint.
        Default is `None` which will use data mean + 0.5 * sigma.

    Returns
    -------
    segmented_image_fixed : `numpy.ndrray`
        The segmented image with brightpoints marked as 3.
    brightpoint_count: `int`
        The number of brightpoints identified in the image.
    granule_count: `int`
        The number of granules identified, after re-classifcation of brightpoint.
    """
    bp_size_limit = (
        0.1  # Approximate max size of a photosphere bright point in square arcsec (see doi 10.3847/1538-4357/aab150)
    )
    bp_pix_upper_limit = bp_size_limit / (resolution**2)
    bp_pix_lower_limit = 4  # Very small bright regions are likley artifacts
    # General flux limit determined by visual inspection.
    if bp_min_flux is None:
        stand_devs = 0.5
        bp_brightness_limit = np.mean(data) + stand_devs * np.std(data)
    else:
        bp_brightness_limit = bp_min_flux
    if len(np.unique(segmented_image)) > 3:
        raise ValueError("segmented_image must have only values of 1, 0 and a 2 (if dim centers marked)")
    segmented_image_fixed = np.copy(segmented_image.astype(float))  # Make type float to enable adding float values
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
    bp_count = 0
    for value in values:
        mask = np.zeros_like(segmented_image)
        mask[labeled_seg == value] = 1
        # Check that is a 1 (white) region.
        if np.sum(np.multiply(mask, segmented_image)) > 0:
            region_size = len(segmented_image_fixed[mask == 1])
            tot_flux = np.sum(data[mask == 1])
            # check that region is small.
            if region_size < bp_pix_upper_limit:
                # Check that region is not *too* small (likley an artifact)
                if region_size > bp_pix_lower_limit:
                    # Check that avg flux very high.
                    if tot_flux / region_size > bp_brightness_limit:
                        segmented_image_fixed[mask == 1] = 3
                        bp_count += 1
    gran_count = len(values) - 1 - bp_count  # Subtract 1 for IG region.
    return segmented_image_fixed, bp_count, gran_count


def segments_overlap_fraction(segment1, segment2):
    """
    Compute the fraction of overlap between two segmented SunPy Maps.

        Designed for comparing output Map from `segment` with other segmentation methods.

    Parameters
    ----------
    segment1: `~sunpy.map.GenericMap`
        Main `~sunpy.map.GenericMap` to compare against. Must have 0 = intergranule, 1 = granule.
    segment2 :`~sunpy.map.GenericMap`
        Comparison `~sunpy.map.GenericMap`. Must have 0 = intergranule, 1 = granule.
        As an example, this could come from a simple segment useing sklearn.cluster.KMeans

    Returns
    -------
    confidence : `float`
        The numeric confidence metric: 0 = no agreement and 1 = complete agreement.
    """
    segment1 = np.array(segment1.data)
    segment2 = np.array(segment2.data)
    total_granules = np.count_nonzero(segment1 == 1)
    total_intergranules = np.count_nonzero(segment1 == 0)
    if total_granules == 0:
        raise ValueError("No granules in `segment1`. It is possible the clustering failed.")
    if total_intergranules == 0:
        raise ValueError("No intergranules in `segment1`. It is possible the clustering failed.")
    granule_agreement_count = 0
    intergranule_agreement_count = 0
    granule_agreement_count = ((segment1 == 1) * (segment2 == 1)).sum()
    intergranule_agreement_count = ((segment1 == 0) * (segment2 == 0)).sum()
    percentage_agreement_granules = granule_agreement_count / total_granules
    percentage_agreement_intergranules = intergranule_agreement_count / total_intergranules
    confidence = np.mean([percentage_agreement_granules, percentage_agreement_intergranules])
    return confidence
