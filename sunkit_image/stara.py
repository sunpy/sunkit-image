from functools import partial

import astropy.time as Time
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map
from astropy.table import QTable
from skimage.filters import median
from skimage.measure import label, regionprops_table
from skimage.morphology import disk, square, white_tophat
from skimage.util import invert

__all__ = ["stara", "get_regions", "plot_sunspots", "get_segs"]


@u.quantity_input
def stara(
    smap,
    circle_radius: u.deg = 100 * u.arcsec,
    median_box: u.deg = 10 * u.arcsec,
    threshold=6000,
    limb_filter: u.percent = None,
):
    """
    A method for automatically detecting sunspots in white-light data using
    morphological operations.

    Parameters
    ----------
    smap : `sunpy.map.GenericMap`
        The map to apply the algorithm to.

    circle_radius : `astropy.units.Quantity`, optional
        The angular size of the structuring element used in the
        `skimage.morphology.white_tophat`. This is the maximum radius of
        detected features.

    median_box : `astropy.units.Quantity`, optional
        The size of the structuring element for the median filter, features
        smaller than this will be averaged out.

    threshold : `int`, optional
        The threshold used for detection, this will be subject to detector
        degradation. The default is a reasonable value for HMI continuum images.

    limb_filter : `astropy.units.Quantity`, optional
        If set, ignore features close to the limb within a percentage of the
        radius of the disk. A value of 10% generally filters out false
        detections around the limb with HMI continuum images.
    """
    data = invert(smap.data)

    # Filter things that are close to limb to reduce false detections
    if limb_filter is not None:
        hpc_coords = sunpy.map.all_coordinates_from_map(smap)
        r = np.sqrt(hpc_coords.Tx**2 + hpc_coords.Ty**2) / (smap.rsun_obs - smap.rsun_obs * limb_filter)
        data[r > 1] = np.nan

    # Median filter to remove detections based on hot pixels
    m_pix = int((median_box / smap.scale[0]).to_value(u.pix))
    med = median(data, square(m_pix), behavior="ndimage")

    # Construct the pixel structuring element
    c_pix = int((circle_radius / smap.scale[0]).to_value(u.pix))
    circle = disk(c_pix / 2)

    finite = white_tophat(med, circle)
    finite[np.isnan(finite)] = 0  # Filter out nans

    return finite > threshold


def get_regions(segmentation, smap):
    """
    Extracts regions from a segmented image and returns a table with their
    properties.

    Parameters
    ----------
    segmentation : `numpy.ndarray`
        A 2D array representing the segmented image, where different regions are marked with different integer labels.

    smap : `sunpy.map.GenericMap`
        The original SunPy map from which the segmented image was derived. This is used to convert pixel coordinates to world coordinates.

    Returns
    -------
    regions : `astropy.table.QTable`
        A table containing the properties of each detected region. The properties include the label, centroid, area, and minimum intensity of each region, as well as the observation time and the heliographic Stonyhurst coordinates of the region center. If no regions are detected, an empty table is returned.
    """
    labelled = label(segmentation)
    if labelled.max() == 0:
        return QTable()

    regions = regionprops_table(labelled, smap.data, properties=["label", "centroid", "area", "min_intensity"])

    regions["obstime"] = Time([smap.date] * regions["label"].size)
    regions["center_coord"] = smap.pixel_to_world(
        regions["centroid-0"] * u.pix,
        regions["centroid-1"] * u.pix,
    ).heliographic_stonyhurst

    return QTable(regions)


def plot_sunspots(segs, maps):
    """
    Plots sunspots detected in solar maps.

    Parameters
    ----------
    seg : list of `numpy.ndarray`
        A list of 2D arrays representing the segmented images, where different regions are marked with different integer labels.

    maps : list of `sunpy.map.GenericMap`
        A list of the original SunPy maps from which the segmented images were derived. These are used to plot the original solar images and to provide the projection for the plots.

    Notes
    -----
    This function creates a new figure and subplot for each map and segmented image pair in the input lists, plots the original solar image, overlays the detected sunspots as contours, and then displays the plots.
    """
    for smap, seg in zip(maps, segs):
        plt.figure()
        ax = plt.subplot(projection=smap)
        smap.plot()
        ax.contour(seg, levels=0)

    plt.show()


def get_segs(maps, limb_filter_value):
    """
    Applies the stara function to each map in the input list with a specified
    limb filter value.

    Parameters
    ----------
    maps : list of `sunpy.map.GenericMap`
        A list of SunPy maps to which the stara function will be applied.

    limb_filter_value : `astropy.units.Quantity`
        The limb filter value to be used in the stara function. This value specifies the percentage of the radius of the disk to ignore features close to the limb.

    Returns
    -------
    segs : list of `numpy.ndarray`
        A list of 2D arrays representing the segmented images resulting from the application of the stara function to each map in the input list. Each segmented image has different regions marked with different integer labels, representing detected sunspots.

    Notes
    -----
    This function uses the functools.partial function to create a new function that behaves like the stara function, but with the limb_filter argument set to limb_filter_value. This new function is then applied to each map in the input list.
    """
    return list(map(partial(stara, limb_filter=limb_filter_value), maps))
