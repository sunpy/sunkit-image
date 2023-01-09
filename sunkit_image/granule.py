"""
This module contains functions that will segment images for 
granule detection.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patheffects as mpe
import matplotlib.colors as colors
import numpy as np
import scipy.ndimage as sndi
import skimage
import sunpy
import sunpy.map
from sklearn.cluster import KMeans as KMeans

__all__ = [
    "segment",
    "get_threshold",
    "trim_intergranules",
    "mark_faculae",
    "segment_kmeans",
    "cross_correlation",
]

def segment(id, data_map, skimage_method, res, plot_intermed=True,
            mark_dim_centers=False, out_dir='output/'):

    """
    Segment optical image of the solar photosphere into tri-value maps
    with 0 = intergranule, 0.5 = faculae, 1 = granule.
    ----------
    Parameters:
        id (string): identifying name of map to be segmented, for file naming
        data_map (SunPy map): SunPy map containing data to segment
        skimage_method (string): skimage thresholding method -
                                options are 'otsu', 'li', 'isodata',
                                'mean', 'minimum', 'yen', 'triangle'
        plot_intermed (True or False): whether to plot and save intermediate
                                data product image
        mark_dim_centers (True or False): whether to mark dim granule centers
                                as a seperate catagory for future exploration
        out_dir (str): Desired directory in which to save intermediate data
                                product image (if plot_intermed = True);
        res (float): Spatial resolution (arcsec/pixel) of the data
    ----------
    Returns:
        segmented_map (SunPy map): SunPy map containing segmentated image
                              (with the original header)
    """

    if type(data_map) != sunpy.map.mapbase.GenericMap:
        raise TypeError('Input must be sunpy map.')

    methods = ['otsu', 'li', 'isodata', 'mean', 'minimum', 'yen', 'triangle']
    if skimage_method not in methods:
        raise TypeError('Method must be one of: ' + str(methods))

    data = data_map.data
    header = data_map.meta

    # apply median filter
    median_filtered = sndi.median_filter(data, size=3)

    # apply threshold
    threshold = get_threshold(median_filtered, skimage_method)

    # initial skimage segmentation
    segmented_image = np.uint8(median_filtered > threshold)

    # fix the extra IGM bits in the middle of granules
    segmented_image_fixed = trim_intergranules(segmented_image,
                                               mark=mark_dim_centers)

    # mark faculae
    segmented_image_markfac = mark_faculae(segmented_image_fixed, data, res)

    if plot_intermed:
        # show pipeline process
        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(14, 13))
        s1 = 16
        s2 = 22
        fig.suptitle('Intermediate Processesing Steps \n', fontsize=s2)

        # define colormap to bring out faculae and dim middles
        col_dict = {0: "black",
                    0.5: "blue",
                    1: "white",
                    1.5: "#ffc406"}
        cmap = colors.ListedColormap([col_dict[x] for x in col_dict.keys()])
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        norm = colors.BoundaryNorm(norm_bins, 4, clip=True)

        im0 = ax0.imshow(data / np.max(data), origin='lower')
        ax0.set_title('Scaled Intensity Data', fontsize=s1)
        plt.colorbar(im0, ax=ax0, shrink=0.8)

        im1 = ax1.imshow(segmented_image, norm=norm, cmap=cmap,
                         interpolation='none', origin='lower')
        ax1.set_title('Initial Thresholding', fontsize=s1)

        im2 = ax2.imshow(segmented_image_fixed, norm=norm, cmap=cmap,
                         interpolation='none', origin='lower')
        if mark_dim_centers:
            ax2.set_title('Dim IG Material Marked', fontsize=s1)
        if not mark_dim_centers:
            ax2.set_title('Extraneous IG Material Removed', fontsize=s1)

        im3 = ax3.imshow(segmented_image_markfac, norm=norm, cmap=cmap,
                         interpolation='none', origin='lower')
        ax3.set_title('Faculae Identified', fontsize=s1)

        plt.tight_layout()

        # rescale axis
        l0, b0, w0, h0 = ax0.get_position().bounds
        newpos = [l0, b0-0.01, w0, h0]
        ax0.set_position(newpos)
        l1, b1, w1, h1 = ax1.get_position().bounds
        newpos = [l1, b0-0.01, w0, h0]
        ax1.set_position(newpos)
        l2, b2, w2, h2 = ax2.get_position().bounds
        newpos = [l0, b2, w0, h0]
        ax2.set_position(newpos)
        l3, b3, w3, h3 = ax3.get_position().bounds
        newpos = [l3, b3, w0, h0]
        ax3.set_position(newpos)

        # add color bar at top
        outline = mpe.withStroke(linewidth=5, foreground='black')
        legax = plt.axes([0.1, 0.1, 0.8, 0.85], alpha=0)
        legax.axis('off')
        if mark_dim_centers:
            labels = ['Granule', 'Intergranule', 'Faculae', 'Dim Centers']
            custom_lines = [lines.Line2D([0], [0], color='white', lw=4,
                                         path_effects=[outline]),
                            lines.Line2D([0], [0], color='black', lw=4),
                            lines.Line2D([0], [0], color="#ffc406", lw=4),
                            lines.Line2D([0], [0], color='blue', lw=4)]
            ncol = 4
        if not mark_dim_centers:
            labels = ['Granule', 'Intergranule', 'Faculae']
            custom_lines = [lines.Line2D([0], [0], color='white', lw=4,
                                         path_effects=[outline]),
                            lines.Line2D([0], [0], color='black', lw=4),
                            lines.Line2D([0], [0], color="#ffc406", lw=4)]
            ncol = 3
        legax.legend(custom_lines, labels, loc='upper center', ncol=ncol,
                     fontsize='x-large')

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except Exception:
                raise OSError('Could not make directory ' + out_dir)

        plt.savefig(out_dir + 'segmentation_plots_' + id + '.png')

    # convert segmentated image back into SunPy map with original header
    segmented_map = sunpy.map.Map(segmented_image_markfac, header)

    return segmented_map

def get_threshold(data, method):
    """
    Get the threshold value using given skimage segmentation type.
    ----------
    Parameters:
        data (numpy array): data to threshold
        method (string): skimage thresholding method - options are 'otsu',
                        'li', 'isodata', 'mean', 'minimum', 'yen', 'triangle'
    ----------
    Returns:
        threshold (float): threshold
    """

    if not type(data) == np.ndarray:
        raise ValueError('Input data must be an array.')

    methods = ['otsu', 'li', 'isodata', 'mean', 'minimum', 'yen', 'triangle']
    if method not in methods:
        raise ValueError('Method must be one of: ' + str(methods))
    if method == 'otsu':  # works ok, but classifies low-flux ganules as IG
        threshold = skimage.filters.threshold_otsu(data)
    elif method == 'li':  # slightly better than otsu
        threshold = skimage.filters.threshold_li(data)
    elif method == 'yen':  # poor - overidentifies IG
        threshold = skimage.filters.threshold_yen(data)
    elif method == 'mean':  # similar to li
        threshold = skimage.filters.threshold_mean(data)
    elif method == 'minimum':  # terrible - identifies almost all as granule
        threshold = skimage.filters.threshold_minimum(data)
    elif method == 'triangle':  # overidentifies IG worse than yen
        threshold = skimage.filters.threshold_triangle(data)
    elif method == 'isodata':  # similar to otsu
        threshold = skimage.filters.threshold_isodata(data)

    return threshold


def trim_intergranules(segmented_image, mark=False):
    """
    Remove the erronous idenfication of intergranule material in the
    middle of granules that pure threshold segmentation produces.
    ----------
    Parameters:
        segmented_image (numpy array): the segmented image containing
                                       incorrect extra intergranules
        mark (bool): if False, remove erronous intergranules. If True,
                     mark them as 0.5 instead (for later examination).
    ----------
    Returns:
        segmented_image_fixed (numpy array): the segmented image without
                                             incorrect extra intergranules
    """

    if len(np.unique(segmented_image)) > 2:
        raise ValueError('segmented_image must have only values of 1 and 0')

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
    ----------
    Parameters:
        segmented_image (numpy array): the segmented image containing
                                incorrect middles
        data (numpy array): the original flux values
        res (float): Spatial resolution (arcsec/pixel) of the data
    ----------
    Returns:
        segmented_image_fixed (numpy array): the segmented image with faculae
                                             marked as 1.5
    """

    fac_size_limit = 2  # max size of a faculae in sqaure arcsec
    fac_pix_limit = fac_size_limit / res
    fac_brightness_limit = np.mean(data) + 0.5 * np.std(data)

    if len(np.unique(segmented_image)) > 3:
        raise ValueError('segmented_image must have only values of 1, 0, ' +
                         'an 0.5 (if dim centers marked)')

    segmented_image = segmented_image
    segmented_image_fixed = np.copy(segmented_image.astype(float))
    labeled_seg = skimage.measure.label(segmented_image + 1, connectivity=2)
    values = np.unique(labeled_seg)
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

    return segmented_image_fixed

def kmeans_segment(data, llambda_axis=-1):
    """kmeans clustering: uses a kmeans algorithm to cluster data,
       in order to independently verify the skimage clustering method
       (e.g using cross_correlation() below).
        ----------
       Parameters:
            data (numpy array): data to be clustered
            llambda_axis (int): index for wavelength, -1 if scalar array.
        ----------
        Returns:
            labels (numpy array): an array of labels, with 0 = granules,
                                  2 = intergranules, 1 = in-between.
            """

    if llambda_axis not in [-1, 2]:
        raise Exception('Wrong data shape. \
        (either scalar or (x, y, llambda) )')
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
        labels = np.reshape(Kmeans(n_clusters, n_init).fit(data),
                            (x_size, y_size))

    # make intergranules 0, granules 1:
    group0_mean = np.mean(data[labels == 0])
    group1_mean = np.mean(data[labels == 1])
    group2_mean = np.mean(data[labels == 2])

    # intergranules
    min_index = np.argmin([group0_mean,
                           group1_mean,
                           group2_mean])
    segmented_map = np.ones(labels.shape)

    segmented_map[[labels[:, :] == min_index][0]] -= 1

    return segmented_map

def cross_correlation(segment1, segment2):
    """
    Return -1 and print a message if the agreement between two
    arrays is low, 0 otherwise. Designed to be used with segment
    and segment_kmeans function.
    ----------
    Parameters:
        segment1 (numpy array): 'main' array to compare the other input
                                 array against.
        segment2 (numpy array): 'other' array (i.e. data
                                 segmented using kmeans).
    ----------
    Returns:
        [label, confidence] (list): where
        label is a label to summarize the confidence metric (int):
            -1: if agreement is low (below 75%)
            0: otherwise
        confidence is the actual confidence metric (float):
            float between 0 and 1 (0 if no agreement,
                                   1 if completely agrees)
    """

    total_granules = np.count_nonzero(segment1 == 1)
    total_intergranules = np.count_nonzero(segment1 == 0)

    if total_granules == 0:
        raise Exception('clustering problematic (no granules found)')

    if total_intergranules == 0:
        raise Exception('clustering problematic (no intergranules found)')

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

    percentage_agreement_granules = \
        granule_agreement_count / total_granules
    percentage_agreement_intergranules = \
        intergranule_agreement_count / total_intergranules
    try:
        confidence = np.mean([percentage_agreement_granules,
                             percentage_agreement_intergranules])
    except TypeError:
        confidence = 0

    if percentage_agreement_granules < 0.75 \
            or percentage_agreement_intergranules < 0.75:
        print('Low agreement with K-Means clustering. \
                         Saved output has low confidence.')
        return [-1, confidence]
    else:
        return [0, confidence]
