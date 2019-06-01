"""
This module contains functions that can be used to enhance the entire solar image.
"""

import numpy as np
import scipy.ndimage as ndimage
import astropy.units as u
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
        raise ValueError ("nsm1 should be less than nsm2")
    
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
    new_image : `sunpy.map`
        Image with loops marked.

    References
    ----------
    * Markus J. Aschwanden, Bart De Pontieu, Eugene A. Katrukha.
      Optimization of Curvi-Linear Tracing Applied to Solar Physics and Biophysics.
    """

    # 1. Supress the background
    image = background_supression(smap.data, zmin, qmed)

    # 2. Bandpass filter
    image = bandpass_filter(image, nsm1, nsm2)
    
    x, y = np.meshgrid(*[np.arange(v.value) for v in smap.dimensions]) * u.pix
    coords = smap.pixel_to_world(x, y).transform_to(frames.Helioprojective)



    # Creating the three starting arrays
    n_s =100
    delta_s = 1
    s_bi = ((np.arange(n_s) - n_s / 2) * delta_s).reshape((-1, 1))
    s_uni = (delta_s * np.arange(n_s)).reshape((-1, 1))

    n_alpha = 180
    alp_l = (np.arange(n_alpha) * (np.pi / n_alpha)).reshape((-1, 1))

    n_r = 30
    r_m = (rmin / (-1 + np.arange(n_r) * (2 / n_r - 1))).reshape((-1, 1))

    loops = []
    ngaps = 3               # Number of empty pixels to denote the end of loop

    # Loops tracing begin
    for i in range(num_loop):

        z_0 = image.max()
        if (z_0 < noise_thresh):
            break
        i_0, j_0 = np.where(image == z_0)

        loop = []

        x_0 = coords[i_0, j_0].Tx.value
        y_0 = coords[i_0, j_0].Ty.value
        loop.append([x_0, y_0])
        
        x_k = x_0
        y_k = y_0

        for sigma in [-1, 1]:               # To deal with both forward and backward pass

            count = 0
            while count < ngaps :
                
                x_k_l = i_0 + s_bi * np.cos(alp_l.T)
                y_k_l = j_0 + s_bi * np.sin(alp_l.T)
                
                # TODO: Write after understanding the updates. Assume we got # Doing the forward pass
                sigma = alpha_k

                alpha_k = 12 # arbitrary for the time being

                beta_0 = alpha_k + np.pi / 2

                x_c = x_k + rmin * np.cos(beta_0)
                y_c = y_k + rmin * np.sin(beta_0)

                x_m = x_k + ((x_c - x_0) / rmin) * r_m
                y_m = y_k + ((y_c - y_0) / rmin) * r_m

                beta_m = beta_0 + sigma * (s_uni / rmin)
                
                x_k_m = x_m - r_m * np.cos(beta_m)
                y_k_m = y_m - r_m * np.sin(beta_m)

                # TODO:Calculate the  rm, the radius of curvature

                rm = 5 # For the time being

                alpha_k_1 = alpha_k + sigma * (delta_s / rm)
                alpha_mid = (alpha_k + alpha_k_1) / 2

                x_k_1 = x_k + delta_s * np.cos(alpha_mid + (1 + sigma) * np.pi / 2)
                y_k_1 = y_k + delta_s * np.sin(alpha_mid + (1 + sigma) * np.pi / 2)

                if image(x_k_1,y_k_1) == 0 : # representation of what to do
                    count += 1 
                
                # Better to convert to pixel values before appending
                loop.append((x_k_1, y_k_1))

                x_k = x_k_1
                y_k = y_k_1
            
            if sigma == -1:
                loop.reverse()
                x_k = x_0
                y_k = y_0
            
        
        # Zero out the loop pixels assuming values are stored in pixels format
        for points in loop:
            image[points[0], points[1]] = 0
         
        loops.append(loop)

    # Return a map with loops marked. What intensity values to be used for marking loops?
    return
