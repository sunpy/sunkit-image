"""
This module contains functions that can be used to enhance the entire solar image.
"""

import numpy as np
import scipy.ndimage as ndimage

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map.maputils as maputils
from sunpy.coordinates import frames

__all__ = [
    "background_supression",
    "bandpass_filter",
    "occult2",
]


def background_supression(image, zmin, qmed=1.0):

    """
    Supresses the background by replacing the pixel intensity values less than `zmin` by product
    of `qmed` and `zmed`, which is the median intensity.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image on which background supression is to be performed.
    zmin : `float`
        The minimum value of intensity which is allowed.
    qmed : `float`
        The scaling factor with which the median is multiplied to fill the values below `zmin`.
        Defaults to 1.0

    Returns
    -------
    new_image : `numpy.ndarray`
        Background supressed image.
    """

    zmed = np.median(image)

    new_image = np.where(image < zmin, qmed * zmed, image)

    return new_image


def bandpass_filter(image, nsm1=1, nsm2=3):

    """
    Applies a band pass filter to the image.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image to be filtered.
    nsm1 : `int`
        Low pass filter boxcar smoothing constant.
        Defaults to 1.
    nsm2 : `int`
        High pass filter boxcar smoothing constant.
        The value of `nsm2` equal to `nsm1 + 1` gives the best enhancement.
        Defaults to 3.

    Returns
    -------
    new_image : `numpy.ndarray`
        Bandpass filtered image.
    """

    if(nsm1 >= nsm2):
        raise ValueError("nsm1 should be less than nsm2")

    return ndimage.uniform_filter(image, nsm1) - ndimage.uniform_filter(image, nsm2)


def occult2(smap, zmin, num_loop, noise_thresh, qmed=1, nsm1=1, nsm2=3, rmin=30, nmax=1000):

    """
    Implements the OCCULT-2 algorithm for loop tracing in solar images.

    Parameters
    ----------
    smap : `sunpy.map`
        Map on which loops are to be detected.
    zmin : `float`
        The minimum value of intensity which is allowed.
    num_loop : `int`
        The maximum number of loops to be detected per image.
    noise_thresh : `float`
        The intensity value below which pixels are considered noisy.
    qmed : `float`
        The scaling factor with which the median is multiplied to fill the values below `zmin`.
        Defaults to 1.0
    nsm1 : `int`
        Low pass filter boxcar smoothing constant.
        Defaults to 1.
    nsm2 : `int`
        High pass filter boxcar smoothing constant.
        The value of `nsm2` equal to `nsm1 + 1` gives the best enhancement.
        Defaults to 3.
    rmin : `int`
        The minimum radius of curvature of the loop to be detected.
        Defaults to 30.
    nmax : `int`
        Maximum number of loops to be detected.
        Defaults to 1000.

    Returns
    -------
    loops : `list`
        A list of all loop where each element is a `astropy.coordinates.SkyCoord` object

    References
    ----------
    * Markus J. Aschwanden, Bart De Pontieu, Eugene A. Katrukha.
      Optimization of Curvi-Linear Tracing Applied to Solar Physics and Biophysics.
    """

    # 1. Supress the background
    image = background_supression(smap.data, zmin, qmed)

    # 2. Bandpass filter
    image = bandpass_filter(image, nsm1, nsm2)

    coords = maputils.all_coordinates_from_map(smap)

    # Creating the three starting arrays

    num_loop_segments = 100  # How to decide this

    # The difference between two loop points
    delta_segment = 1

    segments_bi = ((np.arange(num_loop_segments) - num_loop_segments / 2) * delta_segment).reshape((-1, 1))
    segments_uni = (delta_segment * np.arange(num_loop_segments)).reshape((-1, 1))

    num_ang_segment = 180
    ang_segment = (np.arange(num_ang_segment) * (np.pi / num_ang_segment)).reshape((-1, 1))

    num_radial_segments = 30
    radial_segment = (rmin / (-1 + np.arange(num_radial_segments) * (2 / num_radial_segments - 1))).reshape((-1, 1))

    loops = []  # List of all loops
    ngaps = 3  # Number of empty pixels to denote the end of loop

    # Loops tracing begin
    for _ in range(num_loop):

        z_0 = image.max()  # First point of the loop with maximum intensity

        if (z_0 < noise_thresh):  # Stop loop tracing if maximum value is noise
            break

        i_0, j_0 = np.where(image == z_0)

        loop = []  # To trace a single loop

        # Coordinates of maximum value
        x_0 = coords[i_0, j_0].Tx.value
        y_0 = coords[i_0, j_0].Ty.value
        loop.append([x_0, y_0])

        x_k = x_0   # x_k denotes x-coordinate of kth segment of a loop
        y_k = y_0   # Current loop point

        # x_k_l denotes x-coordinate of kth segment at a particular 'l' angle
        # Same with y-coordinate
        x_k_l = i_0 + segments_bi * np.cos(ang_segment.T)
        y_k_l = j_0 + segments_bi * np.sin(ang_segment.T)

        # TODO: Write after understanding the updates. Assume we got angle_k

        # amgle calculated at kth segment
        angle_k = 12  # arbitrary for the time being

        # angle along proposed centre of curvature
        beta_0 = angle_k + np.pi / 2

        # Coordinates of centre with 'rmin' radius
        x_c = x_k + rmin * np.cos(beta_0)
        y_c = y_k + rmin * np.sin(beta_0)

        for sigma in [-1, 1]:  # To deal with both forward and backward pass

            count = 0  # To make sure loop only finishes after three empty pixels
            while count < ngaps:

                # Loci of centres with radius of curvature as 'radial_segment'
                x_m = x_k + ((x_c - x_0) / rmin) * radial_segment
                y_m = y_k + ((y_c - y_0) / rmin) * radial_segment

                beta_m = beta_0 + sigma * (segments_uni / rmin)

                # TODO: verify if the radius calculation is done for every loop point
                # Not mentioned clearly in the paper but IDL seems to do so.
                x_k_m = x_m - radial_segment * np.cos(beta_m)
                y_k_m = y_m - radial_segment * np.sin(beta_m)

                # TODO:Calculate the  rm, the radius of curvature
                # Current loop point will change

                rm = 5  # For the time being

                angle_k_1 = angle_k + sigma * (delta_segment / rm)
                alpha_mid = (angle_k + angle_k_1) / 2

                x_k_1 = x_k + delta_segment * np.cos(alpha_mid + (1 + sigma) * np.pi / 2)
                y_k_1 = y_k + delta_segment * np.sin(alpha_mid + (1 + sigma) * np.pi / 2)

                if image(x_k_1, y_k_1) == 0:  # representation of what to do
                    count += 1

                loop.append([x_k_1, y_k_1] * u.arcsec)

                x_k = x_k_1
                y_k = y_k_1

            if sigma == -1:
                loop.reverse()
                x_k = x_0
                y_k = y_0

        # Zero out the loop pixels assuming values are stored in pixels format
        for points in loop:
            image[points[0], points[1]] = 0

        loops.append(SkyCoord(loop, frame=frames.Helioprojective))

    return loops
