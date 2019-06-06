"""
This module contains functions that can be used to enhance the entire solar image.
"""

import numpy as np
import scipy.ndimage as ndimage


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


def occult2(smap, zmin, noise_thresh, qmed=1, nsm1=1, nsm2=3, rmin=30, nmax=1000):

    """
    Implements the OCCULT-2 algorithm for loop tracing in solar images.

    Parameters
    ----------
    smap : `sunpy.map`
        Map on which loops are to be detected.
    zmin : `float`
        The minimum value of intensity which is allowed.
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

    ix, iy = np.shape(image)

    # Creating the three starting arrays

    num_loop = 1000  # Maximum number of loops per image

    num_loop_segments = rmin

    width = np.max(nsm2/2-1, 1)  # Width around the loop to be deleted after tracing

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

        loop = []  # To trace a single loop having coordinates of loop points
        angles = []  # To store the angle value for all loop points
        rad_index = []  # To store the radii value of all loop points

        loop.append([i_0, j_0])

        # x_k_l denotes x-coordinate of kth segment at a particular 'l' angle
        # Same with y-coordinate
        x_k_l = loop[-1][0] + np.matmul(segments_bi, np.cos(ang_segment).T)
        y_k_l = loop[-1][1] + np.matmul(segments_bi, np.sin(ang_segment).T)

        x_k_l = np.ceil(x_k_l)  # Converting to pixel values
        y_k_l = np.ceil(y_k_l)

        # Calculate the initial angle of the loop
        angle_k = np.argmax(np.mean(image[x_k_l, y_k_l], axis=0)) * (np.pi / num_ang_segment)
        angles.append(angle_k)

        for sigma in [-1, 1]:  # To deal with both forward and backward pass

            count = 0  # To make sure loop only finishes after three empty pixels
            while count < ngaps:
                # angle along proposed centre of curvature
                beta_0 = angles[-1] + np.pi / 2

                # Coordinates of centre with 'rmin' radius
                x_c = loop[-1][0] + rmin * np.cos(beta_0)
                y_c = loop[-1][1] + rmin * np.sin(beta_0)

                if len(rad_index) != 0:
                    radii = radial_segment[np.max(rad_index[-1] - 1, 0): np.min(rad_index[-1] + 1, num_radial_segments - 1), 0]
                else:
                    radii = radial_segment

                # Loci of centres with radius of curvature as 'radial_segment'
                x_m = loop[-1][0] + ((x_c - loop[-1][0]) / rmin) * radii
                y_m = loop[-1][1] + ((y_c - loop[-1][1]) / rmin) * radii

                beta_m = beta_0 + sigma * (segments_uni / rmin)

                x_k_m = x_m - np.matmul(np.cos(beta_m), radii.T)
                y_k_m = y_m - np.matmul(np.sin(beta_m), radii.T)

                x_k_m = np.ceil(x_k_m)
                y_k_m = np.ceil(y_k_m)

                if len(rad_index) != 0:
                    r_i = np.argmax(np.mean(image[x_k_m, y_k_m], axis=0)) + np.max(rad_index[-1] - 1, 0)
                else:
                    r_i = np.argmax(np.mean(image[x_k_m, y_k_m], axis=0))

                rad_index.append(r_i)

                angles.append(angles[-1] + sigma * (delta_segment / radial_segment[rad_index[-1]]))

                alpha_mid = (angles[-2] + angles[-1]) / 2

                x_k_1 = loop[-1][0] + delta_segment * np.cos(alpha_mid + (1 + sigma) * np.pi / 2)
                y_k_1 = loop[-1][1] + delta_segment * np.sin(alpha_mid + (1 + sigma) * np.pi / 2)

                x_k_1 = np.ceil(x_k_1)
                y_k_1 = np.ceil(y_k_1)

                loop.append([x_k_1, y_k_1])

                if image[np.min(np.max(x_k_1, 0), ix), np.min(np.max(y_k_1, 0), ix)] == 0:
                    count += 1

            if sigma == -1:
                loop.reverse()

        # Zero out the loop pixels assuming values are stored in pixels format
        for points in loop:
            image[points[0], points[1]] = 0

        loops.append(loop)

    return loops
