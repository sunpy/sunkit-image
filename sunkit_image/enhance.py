"""
This module contains functions that can be used to enhance the entire solar image.
"""

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt


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

    if (nsm1 <= 2):
        return image - ndimage.uniform_filter(image, nsm2, mode='nearest')

    if (nsm1 >= 3):
        return ndimage.uniform_filter(image, nsm1, mode='nearest') - ndimage.uniform_filter(image, nsm2, mode='nearest')


def occult2(smap, zmin, noise_thresh, qmed=1, nsm1=1, nsm2=3, rmin=30, nmax=1000):

    """
    Implements the OCCULT-2 algorithm for loop tracing in solar images.

    Parameters
    ----------
    smap : `numpy.ndarray`
        Image on which loops are to be detected.
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
      Entropy, vol. 15, issue 8, pp. 3007-3030
      https://arxiv.org/abs/1307.5046
    """

    fig0 = plt.figure()
    plt.imshow(smap)
    fig0.canvas.set_window_title("Input") 
    # plt.show()


    # 1. Supress the background
    image = background_supression(smap, zmin, qmed)
    
    fig1 = plt.figure()
    plt.imshow(image)
    fig1.canvas.set_window_title("Background") 
    # plt.show()    

    # 2. Bandpass filter
    image = bandpass_filter(image, nsm1, nsm2)
    fig2 = plt.figure()
    plt.imshow(image)
    fig2.canvas.set_window_title("Band pass") 
    # plt.show()

    image = image.T

    ix, iy = np.shape(image)

    # Smoothing the image out at the edges
    image[:, 0:nsm2] = 0
    image[:, iy-nsm2:] = 0
    image[0:nsm2, :] = 0
    image[ix - nsm2:, :] = 0

    smooth = image.T
    
    fig3 = plt.figure()
    plt.imshow(smooth)
    fig3.canvas.set_window_title("Smooth") 
    # plt.show()
    # Creating the three starting arrays

    num_loop = 1000  # Maximum number of loops per image

    num_loop_segments = rmin

    width = max(int(nsm2/2-1), 1)  # Width around the loop to be deleted after tracing

    # The difference between two loop points
    delta_segment = 1

    segments_bi = ((np.arange(num_loop_segments) - num_loop_segments / 2) * delta_segment).reshape((-1, 1))
    segments_uni = (delta_segment * np.arange(num_loop_segments)).reshape((-1, 1))

    num_ang_segment = 181
    ang_segment = (np.arange(num_ang_segment) * (np.pi / num_ang_segment)).reshape((-1, 1))

    num_radial_segments = 30
    radial_segment = (rmin / (-1 + np.arange(num_radial_segments) * (2 / num_radial_segments - 1))).reshape((-1, 1))

    loops = []  # List of all loops
    ngaps = 3  # Number of empty pixels to denote the end of loop

    # Loops tracing begin
    for _ in range(num_loop):

        z_0 = image.max()  # First point of the loop with maximum intensity

        if (z_0 <= noise_thresh):  # Stop loop tracing if maximum value is noise
            break

        max_coords = np.where(image == z_0)
        i_0, j_0 = np.array([max_coords[0][0]]), np.array([max_coords[1][0]])

        loop = []  # To trace a single loop having coordinates of loop points
        angles = []  # To store the angle value for all loop points
        rad_index = []  # To store the index values correspomding to radial_segment of all loop points

        loop.append([i_0, j_0])

        # x_k_l denotes x-coordinate of kth segment at a particular 'l' angle
        # Same with y-coordinate
        flux_max = -10000   # an arbitrary low value
        ang_max = 0
        for ang_ind in range(len(ang_segment)):

            x_k_l = loop[-1][0] + segments_bi * ang_segment[ang_ind]
            y_k_l = loop[-1][1] + segments_bi * ang_segment[ang_ind]
            
            x_k_l = np.int_(np.ceil(x_k_l))  # Converting to pixel values
            y_k_l = np.int_(np.ceil(y_k_l))
            # if np.any(x_k_l < 0) or np.any(x_k_l >= ix):
            #     continue
            # if np.any(y_k_l < 0) or np.any(y_k_l >= iy):
            #     continue
            x_k_l = np.clip(x_k_l, 0, ix-1)
            y_k_l = np.clip(y_k_l, 0, iy-1)
            flux = np.mean(image[x_k_l, y_k_l])
            if flux > flux_max:
                flux_max = flux
                ang_max = ang_ind * (np.pi / num_ang_segment)
        
        # Calculate the initial angle of the loop
        angles.append(ang_max)


        for sigma in [-1, 1]:  # To deal with both forward and backward pass

            count = 0  # To make sure loop only finishes after three empty pixels
            while count < ngaps:
                # angle along proposed centre of curvature
                beta_0 = angles[-1] + np.pi / 2

                # Coordinates of centre with 'rmin' radius
                x_c = loop[-1][0] + rmin * np.cos(beta_0)
                y_c = loop[-1][1] + rmin * np.sin(beta_0)

                if len(rad_index) != 0:
                    r_start = max(rad_index[-1] - 1, 0)
                    r_end = min(rad_index[-1] + 1, num_radial_segments - 1)
                else:
                    r_start = 0
                    r_end = num_radial_segments - 1
                

                flux_rad_max = -10000
                rad_max = 0            
                for rad in range(r_start, r_end + 1):

                    x_m = loop[-1][0] + ((x_c - loop[-1][0]) / rmin) * radial_segment[rad]
                    y_m = loop[-1][1] + ((y_c - loop[-1][1]) / rmin) * radial_segment[rad]

                    beta_m = beta_0 + sigma * (segments_uni / rmin)

                    # It denotes the x coordinate of the point with at a particular radius 'm'
                    # See the paper for these variables
                    x_k_m = x_m - np.cos(beta_m) * radial_segment[rad]
                    y_k_m = y_m - np.sin(beta_m) * radial_segment[rad]

                    x_k_m = np.int_(np.ceil(x_k_m))  # Converting to pixel values
                    y_k_m = np.int_(np.ceil(y_k_m))

                    # if np.any(x_k_m < 0) or np.any(x_k_m >= ix):
                    #     continue
                    # if np.any(y_k_m < 0) or np.any(y_k_m >= iy):
                    #     continue
                    x_k_m = np.clip(x_k_m, 0, ix-1)
                    y_k_m = np.clip(y_k_m, 0, iy-1)
                    flux = np.mean(image[x_k_m, y_k_m])
                    if flux > flux_rad_max:
                        flux_rad_max = flux
                        rad_max = rad

                rad_index.append(rad_max)

                angles.append(angles[-1] + sigma * (delta_segment / radial_segment[rad_index[-1]]))

                alpha_mid = (angles[-2] + angles[-1]) / 2

                x_k_1 = loop[-1][0] + delta_segment * np.cos(alpha_mid + (1 + sigma) * np.pi / 2)
                y_k_1 = loop[-1][1] + delta_segment * np.sin(alpha_mid + (1 + sigma) * np.pi / 2)

                x_k_1 = np.ceil(x_k_1)
                y_k_1 = np.ceil(y_k_1)
                x_k_1 = np.clip(np.int_(x_k_1), 0, ix - 1)
                y_k_1 = np.clip(np.int_(y_k_1), 0, iy - 1)

                loop.append([x_k_1, y_k_1])

                if image[min(max(x_k_1, 0), ix - 1), min(max(y_k_1, 0), iy - 1)] <= 0:
                    count += 1
                else:
                    if count!=0:
                        count = 0
            
            if (count == ngaps):
                loop = loop[:(-ngaps)]
                angles = angles[:(-ngaps)]
                rad_index = rad_index[:(-ngaps)]

            if sigma == -1:
                loop.reverse()
                angles.reverse()
                rad_index.reverse()

        # Zero out the loop pixels
        for points in loop:

            # Range of values to be zeroed out
            ran_x1 = min(max(points[0] - width, np.array([0])), np.array([ix]))
            ran_x2 = min(max(points[0] + width, np.array([0])), np.array([ix]))

            ran_y1 = min(max(points[1] - width, np.array([0])), np.array([iy]))
            ran_y2 = min(max(points[1] + width, np.array([0])), np.array([iy]))

            image[ran_x1[0]:ran_x2[0], ran_y1[0]:ran_y2[0]] = 0

        # fig5 = plt.figure()
        # plt.imshow(image.T)
        # fig5.canvas.set_window_title("Zeroed every") 

        # test = image.T
        # for points in loop:
        #     test[points[1], points[0]] = 10000

        # fig6 = plt.figure()
        # plt.imshow(test)
        # fig6.canvas.set_window_title("Every loop") 
        # plt.show()

        loops.append(loop)
    
    # fig5 = plt.figure()
    # plt.imshow(image)
    # fig5.canvas.set_window_title("Zeroed") 
    # plt.show()

    return loops
