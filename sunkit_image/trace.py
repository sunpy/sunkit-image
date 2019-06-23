"""
This module contains functions that will enhance the trace out structures in an
image.
"""

import matplotlib.pyplot as plt
import numpy as np

from sunkit_image.utils import background_supression, bandpass_filter

__all__ = ["occult2"]


def occult2(smap, zmin, noise_thresh, qmed=1, nsm1=1, nsm2=3, rmin=30, nmax=1000):
    """
    Implements the Oriented Coronal CUrved Loop Tracing (OCCULT-2) algorithm
    for loop tracing in solar images.

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
    `list`
        A list of all loop where each element is a `astropy.coordinates.SkyCoord` object

    References
    ----------
    * Markus J. Aschwanden, Bart De Pontieu, Eugene A. Katrukha.
      Optimization of Curvi-Linear Tracing Applied to Solar Physics and Biophysics.
      Entropy, vol. 15, issue 8, pp. 3007-3030
      https://doi.org/10.3390/e15083007
    """

    # Please ignore the plots for the time being they help me understand whether the initial steps
    # are working correctly or not
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

    # Image is transposed because IDL works column major and python is row major. This is done
    # so that the python and the IDL codes look similar
    image = image.T

    ix, iy = np.shape(image)

    # Smoothing the image out at the edges
    image[:, 0:nsm2] = 0
    image[:, iy - nsm2 :] = 0
    image[0:nsm2, :] = 0
    image[ix - nsm2 :, :] = 0

    smooth = image.T

    fig3 = plt.figure()
    plt.imshow(smooth)
    fig3.canvas.set_window_title("Smooth")
    # plt.show()

    num_loop = nmax  # Maximum number of loops per image

    num_loop_segments = rmin

    width = max(int(nsm2 / 2 - 1), 1)  # Width around the loop to be deleted after tracing

    # The difference between two loop points in pixels
    delta_segment = 1

    # Creating the three starting arrays
    # The loop guiding array. The paper takes it as s_uni_k and s_bi_k. The IDL code has s_loop and s0_loop
    segments_bi = ((np.arange(num_loop_segments) - num_loop_segments / 2) * delta_segment).reshape(
        (-1, 1)
    )
    segments_uni = (delta_segment * np.arange(num_loop_segments)).reshape((-1, 1))

    # angle array with all 180 degrees
    num_ang_segment = 181
    ang_segment = (np.arange(num_ang_segment) * (np.pi / num_ang_segment)).reshape((-1, 1))

    # number of segments in which `rmin` is divided.
    num_radial_segments = 30
    radial_segment = (
        rmin / (-1 + np.arange(num_radial_segments) * (2 / num_radial_segments - 1))
    ).reshape((-1, 1))

    loops = []  # List of all loops
    ngaps = 3  # Number of empty pixels to denote the end of loop

    # Loops tracing begin
    for _ in range(num_loop):

        z_0 = image.max()  # First point of the loop with maximum intensity

        if z_0 <= noise_thresh:  # Stop loop tracing if maximum value is noise
            break

        max_coords = np.where(image == z_0)

        # Since lots of points can have intensity equal to highest so we choose the first point. The other points
        # would be traced as a part of a loop or in the next loop.
        i_0, j_0 = np.array([max_coords[0][0]]), np.array([max_coords[1][0]])

        loop = (
            []
        )  # To trace a single loop having coordinates of loop points. Each entry is a point having x and y coordinate
        angles = []  # To store the angle value for all loop points
        rad_index = (
            []
        )  # To store the index values correspomding to radial_segment of all loop points

        # adding the first loop point
        loop.append([i_0, j_0])

        # x_k_l denotes x-coordinate of kth segment at a particular 'l' angle
        # Same with y-coordinate. See eqn 13, 14 in the paper
        x_k_l = loop[-1][0] + np.matmul(segments_bi, np.cos(ang_segment).T)
        y_k_l = loop[-1][1] + np.matmul(segments_bi, np.sin(ang_segment).T)

        x_k_l = np.ceil(x_k_l)  # Converting to pixel values
        y_k_l = np.ceil(y_k_l)
        x_k_l = np.clip(
            np.int_(x_k_l), 0, ix - 1
        )  # Making sure every value is between the valid range.
        y_k_l = np.clip(np.int_(y_k_l), 0, iy - 1)

        # See equation 15 of the paper
        angle_k = np.argmax(np.mean(image[x_k_l, y_k_l], axis=0)) * (np.pi / num_ang_segment)

        angles.append(angle_k)

        for sigma in [-1, 1]:  # To deal with both forward and backward pass

            count = 0  # To make sure loop only finishes after `ngap` empty pixels
            while count < ngaps:

                # angle along proposed centre of curvature
                beta_0 = angles[-1] + np.pi / 2

                # Coordinates of centre with 'rmin' radius. See eqn 17, 18
                x_c = loop[-1][0] + rmin * np.cos(beta_0)
                y_c = loop[-1][1] + rmin * np.sin(beta_0)

                # Range in which to need to search for the radius of the new point. See line 163-171 in the IDL code
                if len(rad_index) != 0:
                    radii = radial_segment[
                        max(rad_index[-1] - 1, 0) : min(rad_index[-1] + 2, num_radial_segments), :
                    ].T
                else:
                    radii = radial_segment.T

                # Loci of centres with radius of curvature as 'radial_segment'. See eqn 19, 20
                x_m = loop[-1][0] + ((x_c - loop[-1][0]) / rmin) * radii
                y_m = loop[-1][1] + ((y_c - loop[-1][1]) / rmin) * radii

                # See eqn 21
                beta_m = beta_0 + sigma * (segments_uni / rmin)

                # See eqn 22, 23
                x_k_m = x_m - np.matmul(np.cos(beta_m), radii)
                y_k_m = y_m - np.matmul(np.sin(beta_m), radii)

                x_k_m = np.ceil(x_k_m)
                y_k_m = np.ceil(y_k_m)
                x_k_m = np.clip(np.int_(x_k_m), 0, ix - 1)
                y_k_m = np.clip(np.int_(y_k_m), 0, iy - 1)

                # See eqn 24
                if len(rad_index) != 0:
                    r_i = np.argmax(np.mean(image[x_k_m, y_k_m], axis=0)) + max(
                        rad_index[-1] - 1, 0
                    )
                else:
                    r_i = np.argmax(np.mean(image[x_k_m, y_k_m], axis=0))

                rad_index.append(r_i)

                # See eqn 25
                angles.append(angles[-1] + sigma * (delta_segment / radial_segment[rad_index[-1]]))

                # See eqn 26
                alpha_mid = (angles[-2] + angles[-1]) / 2

                # See eqn 27,28
                # This is what is mentioned in the paper.
                # x_k_1 = loop[-1][0] + delta_segment * np.cos(alpha_mid + (1 + sigma) * np.pi / 2)
                # y_k_1 = loop[-1][1] + delta_segment * np.sin(alpha_mid + (1 + sigma) * np.pi / 2)

                # This was done in the IDL. Doing this leads to more number of repetitive points and
                # on the test image more number of loops but each loop contains only one point.
                if sigma == -1:
                    idir = 1
                else:
                    idir = 0
                x_k_1 = loop[-1][0] + delta_segment * np.cos(alpha_mid + idir * np.pi)
                y_k_1 = loop[-1][1] + delta_segment * np.sin(alpha_mid + idir * np.pi)

                x_k_1 = np.ceil(x_k_1)
                y_k_1 = np.ceil(y_k_1)
                x_k_1 = np.clip(np.int_(x_k_1), 0, ix - 1)
                y_k_1 = np.clip(np.int_(y_k_1), 0, iy - 1)

                loop.append([x_k_1, y_k_1])

                if (
                    image[min(max(x_k_1, 0), ix - 1), min(max(y_k_1, 0), iy - 1)] <= 0
                ):  # To check whether the detected point is valid
                    count += 1
                else:  # If the point is valid but somewhere during the trace we encountered some points which were not valid but not in succession
                    # So we clear our `count` if we get a valid point after some non valid ones.
                    if count != 0:
                        count = 0

            if (
                count == ngaps
            ):  # The loop is terminated but the last `ngap` points were not valid so we remove them
                loop = loop[:(-ngaps)]
                angles = angles[:(-ngaps)]
                rad_index = rad_index[:(-ngaps)]

            if (
                sigma == -1
            ):  # After one direction of trace is done we reverse our points and start again in the next
                loop.reverse()
                angles.reverse()
                rad_index.reverse()

        # Zero out the loop pixels around the loop
        for points in loop:

            # Range of values to be zeroed out around the loop points
            ran_x1 = min(max(points[0] - width, np.array([0])), np.array([ix]))
            ran_x2 = min(max(points[0] + width, np.array([0])), np.array([ix]))

            ran_y1 = min(max(points[1] - width, np.array([0])), np.array([iy]))
            ran_y2 = min(max(points[1] + width, np.array([0])), np.array([iy]))

            image[ran_x1[0] : ran_x2[0], ran_y1[0] : ran_y2[0]] = 0

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
        if len(loop) <= 1:  # A loop detected having only one point is not valid
            continue

        # Add the traced loop to the list of loops
        loops.append(loop)

    # fig5 = plt.figure()
    # plt.imshow(image)
    # fig5.canvas.set_window_title("Zeroed")
    # plt.show()

    return loops
