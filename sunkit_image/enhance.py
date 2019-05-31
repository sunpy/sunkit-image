"""
This module contains functions that can be used to enhance the entire solar image.
"""

import numpy as np
import scipy.ndimage as ndimage

__all__ = [
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
        raise ValueError ("nsm1 should be less than nsm2")
    
    return ndimage.uniform_filter(image, nsm1) - ndimage.uniform_filter(image, nsm2)


def occult2(image, zmin, qmed=1, nsm1=1, nsm2=3, rmin=30, nmax=1000):
    
    """
    Implements the OCCULT-2 algorithm for loop tracing in solar images.

    Parameters
    ----------
    image : `numpy.ndarray` 
        Image on which loops are to be detected.
    zmin : `float`
        The minimum value of intensity which is allowed.
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
    new_image : `numpy.ndarray`
        Bandpass filtered image.

    References
    ----------
    * Markus J. Aschwanden, Bart De Pontieu, Eugene A. Katrukha.
      Optimization of Curvi-Linear Tracing Applied to Solar Physics and Biophysics.
    """

    # 1. Supress the background
    image = background_supression(image, zmin, qmed)

    # 2. Bandpass filter
    image = bandpass_filter(image, nsm1, nsm2)

    # 3. Start loop tracing
    