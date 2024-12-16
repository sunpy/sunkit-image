"""
This module contains an implementation of the Sunspot Tracking And Recognition
Algorithm (STARA).
"""

import numpy as np
from skimage.filters import median
from skimage.morphology import disk, white_tophat
from skimage.util import invert

import astropy.units as u

import sunpy.map

__all__ = ["stara"]


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
        detected features. By default, this is set to 100 arcseconds.
    median_box : `astropy.units.Quantity`, optional
        The size of the structuring element for the median filter, features
        smaller than this will be averaged out. The default value is 10 arcseconds.
    threshold : `int`, optional
        The threshold used for detection, this will be subject to detector
        degradation. The default value of 6000, is a reasonable value for HMI continuum
        images.
    limb_filter : `astropy.units.Quantity`, optional
        If set, ignore features close to the limb within a percentage of the
        radius of the disk. A value of 10% generally filters out false
        detections around the limb with HMI continuum images.

    Returns
    -------
    `numpy.ndarray`
        A 2D boolean array of the same shape as the input solar map. Each element in the array
        represents a pixel in the solar map, and its value is `True` if the corresponding pixel
        is identified as part of a sunspot (based on the specified threshold), and `False` otherwise.

    References
    ----------
    * Fraser Watson and Fletcher Lyndsay
      "Automated sunspot detection and the evolution of sunspot magnetic fields during solar cycle 23"
      Proceedings of the International Astronomical Union, vol. 6, no. S273, pp. 51-55, 2010. (doi:10.1017/S1743921311014992)[https://doi.org/10.1017/S1743921311014992]
    """
    data = invert(smap.data)

    # Filter things that are close to limb to reduce false detections
    if limb_filter is not None:
        hpc_coords = sunpy.map.all_coordinates_from_map(smap)
        r = np.sqrt(hpc_coords.Tx**2 + hpc_coords.Ty**2) / (smap.rsun_obs - smap.rsun_obs * limb_filter)
        data[r > 1] = np.nan

    # Median filter to remove detections based on hot pixels
    m_pix = int((median_box / smap.scale[0]).to_value(u.pix))

    # Need to account for https://github.com/scikit-image/scikit-image/pull/7566/files
    import skimage
    if skimage.__version__ < "0.25.0":
        from skimage.morphology import square
        function = square(m_pix)
    else:
        from skimage.morphology import footprint_rectangle
        function = footprint_rectangle((m_pix, m_pix))
    med = median(data, function, behavior="ndimage")

    # Construct the pixel structuring element
    c_pix = int((circle_radius / smap.scale[0]).to_value(u.pix))
    circle = disk(c_pix / 2)

    finite = white_tophat(med, circle)
    finite[np.isnan(finite)] = 0

    return finite > threshold
