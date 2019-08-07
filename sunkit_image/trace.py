"""
This module contains functions that will enhance the trace out structures in an
image.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from sunkit_image.utils import background_supression, bandpass_filter

__all__ = ["occult2"]


def occult2(smap, zmin, qthresh, lmin, qmed=1, nsm1=1, nsm2=3, rmin=30, nmax=1000):
    """
    Implements the Oriented Coronal CUrved Loop Tracing (OCCULT-2) algorithm
    for loop tracing in solar images.

    Parameters
    ----------
    smap : `numpy.ndarray`
        Image on which loops are to be detected.
    zmin : `float`
        The minimum value of intensity which is allowed.
    qthresh : `float`
        The scaling factor with which the median is multiplied to find the noise threshold
        in the image.
    lmin : `int`
        The length of the smallest loop.
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
    image[:, iy - nsm2:] = 0
    image[0:nsm2, :] = 0
    image[ix - nsm2:, :] = 0

    smooth = image.T

    fig3 = plt.figure()
    plt.imshow(smooth)
    fig3.canvas.set_window_title("Smooth")
    # plt.show()

    # Calculating the noise threshold of the image
    noise_thresh = np.median(image[image > 0]) * qthresh

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

    xloops = []  # List of containing x-coordinate of all loops
    yloops = []  # List of containing y-coordinate of all loops
    loops = []
    ngaps = 1  # Number of empty pixels to denote the end of loop

    residual = np.where(image > 0, image, 0)

    # Loops tracing begin
    for _ in range(num_loop):

        z_0 = residual.max()  # First point of the loop with maximum intensity

        if z_0 <= noise_thresh:  # Stop loop tracing if maximum value is noise
            break

        max_coords = np.where(residual == z_0)

        # Since lots of points can have intensity equal to highest so we choose the first point.
        # The other points would be traced as a part of a loop or in the next loop.
        i_0, j_0 = np.array([max_coords[0][0]]), np.array([max_coords[1][0]])

        loop = (
            []
        )  # To trace a single loop having coordinates of loop points. Each entry is a point having x and y coordinate
        angles = []  # To store the angle value for all loop points
        rad_index = []  # To store the index values correspomding to radial_segment of all loop points

        # adding the first loop point
        loop.append([i_0, j_0])

        for sigma in [-1, 1]:  # To deal with both forward and backward pass

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

                if residual[x_k_1, y_k_1] <= 0:  # To check whether the detected point is valid
                    count += 1
                else:  # If the point is valid but somewhere during the trace we encountered some
                    # points which were not valid but not in succession
                    # So we clear our `count` if we get a valid point after some non valid ones.
                    if count != 0:
                        count = 0

                # if x_k_1 != loop[-1][0] or y_k_1 != loop[-1][1]:
                loop.append([x_k_1, y_k_1])

            if (
                count == ngaps
            ):  # The loop is terminated but the last `ngap` points were not valid so we remove them
                loop = loop[:(-ngaps)]
                angles = angles[:(-ngaps)]
                rad_index = rad_index[:(-ngaps)]

            if (
                sigma == -1
            ):  # After one direction of trace is done we reverse our points and start again in the
                # next
                loop.reverse()
                angles.reverse()
                rad_index.reverse()

        xloop = []
        yloop = []
        # Zero out the loop pixels around the loop
        for points in loop:

            # Range of values to be zeroed out around the loop points
            # ran_x1 = min(max(points[0] - width, np.array([0])), np.array([ix]))
            # ran_x2 = min(max(points[0] + width, np.array([0])), np.array([ix]))

            # ran_y1 = min(max(points[1] - width, np.array([0])), np.array([iy]))
            # ran_y2 = min(max(points[1] + width, np.array([0])), np.array([iy]))

            xloop.append(points[0])
            yloop.append(points[1])

            i0 = min(max(int(points[0]), 0), ix-1)
            i3 = max(int(i0 - width), 0)
            i4 = min(int(i0 + width), ix - 1)
            j0 = min(max(int(points[1]), 0), iy-1)
            j3 = max(int(j0 - width), 0)
            j4 = min(int(j0 + width), iy - 1)

            image[i3:i4, j3:j4] = 0

            # image[ran_x1[0]: ran_x2[0], ran_y1[0]: ran_y2[0]] = 0

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
        # if len(loop) <= 1:  # A loop detected having only one point is not valid
        #     continue

        num_points = len(loop)
        lengths = np.zeros((num_points,), dtype=float)

        if num_points > 2:
            for i in range(1, num_points):
                lengths[i] = lengths[i - 1] + np.sqrt(
                    (loop[i][0] - loop[i - 1][0]) ** 2 + (loop[i][1] - loop[i - 1][1]) ** 2
                )

        looplen = lengths[-1]
        ns = max(int(looplen), 3)
        ss = np.arange(ns)

        xloop = np.array(xloop).reshape((-1,))
        yloop = np.array(yloop).reshape((-1,))
        lengths = np.array(lengths).reshape((-1,))

        if (looplen >= lmin and len(xloop) > 1):
            reso = 1
            nn = np.ceil(ns / reso)
            ii = np.arange(nn) * reso
            interfunc = interpolate.interp1d(lengths, xloop, fill_value="extrapolate")
            xx = interfunc(ii)
            interfunc = interpolate.interp1d(lengths, yloop, fill_value="extrapolate")
            yy = interfunc(ii)

            # Add the traced loop to the list of loops
            xloops.append(xx)
            yloops.append(yy)
            loops.append(loop)

    # fig5 = plt.figure()
    # plt.imshow(image)
    # fig5.canvas.set_window_title("Zeroed")
    # plt.show()

    return xloops, yloops, loops


def occult(image1, nsm1, rmin, lmin, nstruc, nloop, ngap, qthresh1, qthresh2):
    reso = 1
    step = 1
    nloopmax = 10000
    npmax = 2000
    nsm2 = nsm1+2
    nlen = rmin
    na = 180
    nb = 30

    s_loop = step * np.arange(nlen)
    s0_loop = step * (np.arange(nlen) - nlen // 2)
    wid = max(nsm2 // 2 - 1, 1)
    looplen = 0

    # BASE LEVEL
    zmed = np.median(image1[image1 > 0])
    image1 = np.where(image1 < zmed, zmed * qthresh1, image1)

    # HIGHPASS FILTER_
    image2 = bandpass_filter(image1, nsm1, nsm2)
    nx, ny = image2.shape

    # ERASE BOUNDARIES ZONES (SMOOTHING EFFECTS)
    image2[:, 0:nsm2] = 0
    image2[:, ny - nsm2:] = 0
    image2[0:nsm2, :] = 0
    image2[nx - nsm2:, :] = 0

    # NOISE THRESHOLD
    zmed = np.median(image2[image2 > 0])
    thresh = zmed * qthresh2

    # LOOP TRACING START AT MAXIMUM FLUX POSITION
    iloop = 0
    residual = np.where(image2 > 0, image2, 0)
    iloop_nstruc = np.zeros((nstruc,))
    loop_len = np.zeros((nloopmax,))

    for istruc in range(0, nstruc):
        zstart = residual.max()
        if zstart <= thresh:  # goto: end_trace
            break
        max_coords = np.where(residual == zstart)
        istart, jstart = max_coords[0][0], max_coords[1][0]

        # TRACING LOOP STRUCTURE STEPWISE
        ip = 0
        ndir = 2
        for idir in range(0, ndir):
            xl = np.zeros((npmax + 1,))
            yl = np.zeros((npmax + 1,))
            zl = np.zeros((npmax + 1,))
            al = np.zeros((npmax + 1,))
            ir = np.zeros((npmax + 1,))
            if idir == 0:
                sign_dir = +1
            if idir == 1:
                sign_dir = -1

            # INITIAL DIRECTION FINDING
            xl[0] = istart
            yl[0] = jstart
            zl[0] = zstart
            alpha = np.pi * np.arange(na, dtype=float) / float(na)
            flux_max = 0
            for ia in range(0, na):
                x_ = xl[0] + s0_loop * np.cos(alpha[ia])
                y_ = yl[0] + s0_loop * np.sin(alpha[ia])
                ix = np.int_(np.ceil(x_))  # int(x_ + 0.5)
                iy = np.int_(np.ceil(y_))  # int(y_ + 0.5)
                ix = np.clip(ix, 0, nx - 1)
                iy = np.clip(iy, 0, ny - 1)
                flux_ = residual[ix, iy]
                flux = np.sum(np.maximum(flux_, 0.)) / float(nlen)
                if flux > flux_max:
                    flux_max = flux
                    al[0] = alpha[ia]
                    x_lin = x_
                    y_lin = y_


            # CURVATURE RADIUS
            xx_curv = np.zeros((nlen, nb, npmax))
            yy_curv = np.zeros((nlen, nb, npmax))
            for ip in range(0, npmax):

                if ip == 0:
                    ib1 = 0
                    ib2 = nb-1

                if ip >= 1:
                    ib1 = int(max(ir[ip] - 1, 0))
                    ib1 = int(min(ir[ip] + 1, nb-1))

                beta0 = al[ip] + np.pi / 2
                xcen = xl[ip] + rmin * np.cos(beta0)
                ycen = yl[ip] + rmin * np.sin(beta0)

                flux_max = 0
                for ib in range(ib1, ib2 + 1):
                    rad_i = rmin / (-1. + 2. * float(ib) / float(nb - 1))
                    xcen_i = xl[ip] + (xcen - xl[ip]) * (rad_i / rmin)
                    ycen_i = yl[ip] + (ycen - yl[ip]) * (rad_i / rmin)
                    beta_i = beta0 + sign_dir * s_loop / rad_i
                    x_ = xcen_i - rad_i * np.cos(beta_i)
                    y_ = ycen_i - rad_i * np.sin(beta_i)
                    ix = np.int_(np.ceil(x_))  # int(x_ + 0.5)
                    iy = np.int_(np.ceil(y_))  # int(y_ + 0.5)
                    ix = np.clip(ix, 0, nx - 1)
                    iy = np.clip(iy, 0, ny - 1)
                    flux_ = residual[ix, iy]
                    flux = np.sum(np.maximum(flux_, 0.)) / float(nlen)
                    if idir == 1:
                        xx_curv[:, ib, ip] = x_
                        yy_curv[:, ib, ip] = y_
                    if flux > flux_max:
                        flux_max = flux
                        al[ip + 1] = al[ip] + sign_dir * (step / rad_i)
                        ir[ip+1] = ib
                        al_mid = (al[ip]+al[ip+1]) / 2.
                        xl[ip+1] = xl[ip] + step * np.cos(al_mid + np.pi * idir)
                        yl[ip+1] = yl[ip] + step * np.sin(al_mid + np.pi * idir)
                        ix_ip = min(max(int(xl[ip + 1] + 0.5), 0), nx - 1)
                        iy_ip = min(max(int(yl[ip + 1] + 0.5), 0), ny - 1)
                        zl[ip + 1] = residual[ix_ip, iy_ip]
                        if ip == 0:
                            x_curv = x_
                            y_curv = y_

                iz1 = max((ip + 1 - ngap), 0)
                if np.max(zl[iz1:ip+2]) <= 0:
                    break  # goto endsegm

            # ENDSEGM

            # RE-ORDERING LOOP COORDINATES
            if idir == 0:
                xloop = np.flip(xl[0:ip+1])
                yloop = np.flip(yl[0:ip+1])
                zloop = np.flip(zl[0:ip+1])
            if idir == 1:
                xloop = np.concatenate([xloop, xl[1:ip+1]])
                yloop = np.concatenate([yloop, yl[1:ip+1]])
                zloop = np.concatenate([zloop, zl[1:ip+1]])
        ind = np.logical_and(xloop != 0, yloop != 0)
        nind = np.sum(ind)
        looplen = 0
        if nind > 1:
            # skip_struct
            xloop = xloop[ind]
            yloop = yloop[ind]
            zloop = zloop[ind]

            if iloop >= nloopmax:
                break  # end_trace

            np1 = len(xloop)
            s = np.zeros((np1))
            looplen = 0
            if np1 >= 2:
                for ip in range(1, np1):
                    s[ip] = s[ip - 1] + np.sqrt((xloop[ip] - xloop[ip - 1]) ** 2 + (yloop[ip] - yloop[ip - 1]) ** 2)
            looplen = s[np1-1]
            ns = max(int(looplen), 3)
            ss = np.arange(ns)

        # SKIP STRUCT
        if (looplen >= lmin):
            nn = int(ns / reso + 0.5)
            ii = np.arange(nn) * reso
            interfunc = interpolate.interp1d(s, xloop, fill_value="extrapolate")
            xx = interfunc(ii)
            interfunc = interpolate.interp1d(s, yloop, fill_value="extrapolate")
            yy = interfunc(ii)
            interfunc = interpolate.interp1d(s, zloop, fill_value="extrapolate")
            ff = interfunc(ii)

            x_rsun = xx
            y_rsun = yy
            s_rsun = ii

            loopnum = np.ones((nn)) * iloop
            loop = np.c_[loopnum, xx, yy, ff, ii]

            if iloop == 0:
                loopfile = loop
            if iloop >= 1:
                loopfile = np.r_[loopfile, loop]
            iloop_nstruc[istruc] = iloop
            loop_len[iloop] = looplen
            iloop += 1

        # TEST DISPLAY

        # ERASE LOOP IN RESIDUAL IMAGE
        i3 = max(istart - wid, 0)
        i4 = min(istart + wid, nx - 1)
        j3 = max(jstart - wid, 0)
        j4 = min(jstart + wid, ny - 1)
        residual[i3:i4, j3:j4] = 0.
        nn = len(xloop)
        for iss in range(0, nn):
            i0 = min(max(int(xloop[iss]), 0), nx-1)
            i3 = max(int(i0 - wid), 0)
            i4 = min(int(i0 + wid), nx - 1)
            j0 = min(max(int(yloop[iss]), 0), ny-1)
            j3 = max(int(j0 - wid), 0)
            j4 = min(int(j0 + wid), ny - 1)
            residual[i3:i4, j3:j4] = 0.

    # END_TRACE
    fluxmin = np.min(image1)
    fluxmax = np.max(image1)

    return loopfile, image2


def reordering_loop(xl, yl, zl, ip, idir, xloop, yloop, zloop):
    if idir == 0:
        xloop = np.flip(xl[0:ip+1])
        yloop = np.flip(yl[0:ip+1])
        zloop = np.flip(zl[0:ip+1])
    if idir == 1:
        xloop = np.concatenate([xloop, xl[1:ip+1]])
        yloop = np.concatenate([yloop, yl[1:ip+1]])
        zloop = np.concatenate([zloop, zl[1:ip+1]])
    
    return (xloop, yloop, zloop)


def erase_loop_in_residual(istart, jstart, wid, nx, ny, residual, xloop, yloop):

    i3 = max(istart - wid, 0)
    i4 = min(istart + wid, nx - 1)
    j3 = max(jstart - wid, 0)
    j4 = min(jstart + wid, ny - 1)
    residual[i3:i4, j3:j4] = 0.
    nn = len(xloop)
    for iss in range(0, nn):
        i0 = min(max(int(xloop[iss]), 0), nx-1)
        i3 = max(int(i0 - wid), 0)
        i4 = min(int(i0 + wid), nx - 1)
        j0 = min(max(int(yloop[iss]), 0), ny-1)
        j3 = max(int(j0 - wid), 0)
        j4 = min(int(j0 + wid), ny - 1)
        residual[i3:i4, j3:j4] = 0.
    
    return residual


def loop_add(ns, reso, s, xloop, yloop, zloop, iloop, iloop_nstruc, istruc, loop_len, looplen, loopfile=None):
    nn = int(ns / reso + 0.5)
    ii = np.arange(nn) * reso
    interfunc = interpolate.interp1d(s, xloop, fill_value="extrapolate")
    xx = interfunc(ii)
    interfunc = interpolate.interp1d(s, yloop, fill_value="extrapolate")
    yy = interfunc(ii)
    interfunc = interpolate.interp1d(s, zloop, fill_value="extrapolate")
    ff = interfunc(ii)

    x_rsun = xx
    y_rsun = yy
    s_rsun = ii

    loopnum = np.ones((nn)) * iloop
    loop = np.c_[loopnum, xx, yy, ff, ii]

    if iloop == 0:
        loopfile = loop
    if iloop >= 1:
        loopfile = np.r_[loopfile, loop]
    iloop_nstruc[istruc] = iloop
    loop_len[iloop] = looplen
    iloop += 1

    return loopfile, iloop, loop_len, iloop_nstruc


def initial_direction_finding(xl, yl, s0_loop, na, alpha, residual, nx, ny, al, nlen):
    flux_max = 0
    for ia in range(0, na):
        x_ = xl[0] + s0_loop * np.cos(alpha[ia])
        y_ = yl[0] + s0_loop * np.sin(alpha[ia])
        ix = np.int_(np.ceil(x_))  # int(x_ + 0.5)
        iy = np.int_(np.ceil(y_))  # int(y_ + 0.5)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        flux_ = residual[ix, iy]
        flux = np.sum(np.maximum(flux_, 0.)) / float(nlen)
        if flux > flux_max:
            flux_max = flux
            al[0] = alpha[ia]
            x_lin = x_
            y_lin = y_
    
    return al


def curvature_radius(ib1, ib2, rmin, xl, yl, zl, nb,
                     al, ip, sign_dir, s_loop, residual,
                     nx, ny, nlen, idir, step, ir, xx_curv, yy_curv):

    beta0 = al[ip] + np.pi / 2
    xcen = xl[ip] + rmin * np.cos(beta0)
    ycen = yl[ip] + rmin * np.sin(beta0)
    flux_max = 0
    for ib in range(ib1, ib2 + 1):
        rad_i, flux, x_, y_ = find_flux(rmin, nb, ib, xl, yl, xcen, ycen, ip, beta0, sign_dir, s_loop, residual, nx, ny, nlen)
        if idir == 1:
            xx_curv[:, ib, ip] = x_
            yy_curv[:, ib, ip] = y_
        if flux > flux_max:
            flux_max = flux
            al[ip + 1] = al[ip] + sign_dir * (step / rad_i)
            ir[ip+1] = ib
            al_mid = (al[ip]+al[ip+1]) / 2.
            xl[ip+1] = xl[ip] + step * np.cos(al_mid + np.pi * idir)
            yl[ip+1] = yl[ip] + step * np.sin(al_mid + np.pi * idir)
            ix_ip = min(max(int(xl[ip + 1] + 0.5), 0), nx - 1)
            iy_ip = min(max(int(yl[ip + 1] + 0.5), 0), ny - 1)
            zl[ip + 1] = residual[ix_ip, iy_ip]
            if ip == 0:
                x_curv = x_
                y_curv = y_
    
    return xl, yl, zl, al


def find_flux(rmin, nb, ib, xl, yl, xcen, ycen, ip, beta0, sign_dir, s_loop, residual, nx, ny, nlen):
    rad_i = rmin / (-1. + 2. * float(ib) / float(nb - 1))
    xcen_i = xl[ip] + (xcen - xl[ip]) * (rad_i / rmin)
    ycen_i = yl[ip] + (ycen - yl[ip]) * (rad_i / rmin)
    beta_i = beta0 + sign_dir * s_loop / rad_i
    x_ = xcen_i - rad_i * np.cos(beta_i)
    y_ = ycen_i - rad_i * np.sin(beta_i)
    ix = np.int_(np.ceil(x_))  # int(x_ + 0.5)
    iy = np.int_(np.ceil(y_))  # int(y_ + 0.5)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    flux_ = residual[ix, iy]
    flux = np.sum(np.maximum(flux_, 0.)) / float(nlen)

    return rad_i, flux, x_, y_


def occult3(image1, nsm1, rmin, lmin, nstruc, nloop, ngap, qthresh1, qthresh2):
    reso = 1
    step = 1
    nloopmax = 10000
    npmax = 2000
    nsm2 = nsm1+2
    nlen = rmin
    na = 180
    nb = 30

    s_loop = step * np.arange(nlen)
    s0_loop = step * (np.arange(nlen) - nlen // 2)
    wid = max(nsm2 // 2 - 1, 1)
    looplen = 0

    # BASE LEVEL
    zmed = np.median(image1[image1 > 0])
    image1 = np.where(image1 < zmed, zmed * qthresh1, image1)

    # HIGHPASS FILTER_
    image2 = bandpass_filter(image1, nsm1, nsm2)
    nx, ny = image2.shape

    # ERASE BOUNDARIES ZONES (SMOOTHING EFFECTS)
    image2[:, 0:nsm2] = 0
    image2[:, ny - nsm2:] = 0
    image2[0:nsm2, :] = 0
    image2[nx - nsm2:, :] = 0

    # NOISE THRESHOLD
    zmed = np.median(image2[image2 > 0])
    thresh = zmed * qthresh2

    # LOOP TRACING START AT MAXIMUM FLUX POSITION
    iloop = 0
    residual = np.where(image2 > 0, image2, 0)
    iloop_nstruc = np.zeros((nstruc,))
    loop_len = np.zeros((nloopmax,))

    for istruc in range(0, nstruc):
        zstart = residual.max()
        if zstart <= thresh:  # goto: end_trace
            break
        max_coords = np.where(residual == zstart)
        istart, jstart = max_coords[0][0], max_coords[1][0]

        # TRACING LOOP STRUCTURE STEPWISE
        ip = 0
        ndir = 2
        for idir in range(0, ndir):
            xl = np.zeros((npmax + 1,))
            yl = np.zeros((npmax + 1,))
            zl = np.zeros((npmax + 1,))
            al = np.zeros((npmax + 1,))
            ir = np.zeros((npmax + 1,))
            if idir == 0:
                sign_dir = +1
            if idir == 1:
                sign_dir = -1

            # INITIAL DIRECTION FINDING
            xl[0] = istart
            yl[0] = jstart
            zl[0] = zstart
            alpha = np.pi * np.arange(na, dtype=float) / float(na)

            al = initial_direction_finding(xl, yl, s0_loop, na, alpha, residual, nx, ny, al, nlen)
            
            # CURVATURE RADIUS
            xx_curv = np.zeros((nlen, nb, npmax))
            yy_curv = np.zeros((nlen, nb, npmax))
            for ip in range(0, npmax):

                if ip == 0:
                    ib1 = 0
                    ib2 = nb-1

                if ip >= 1:
                    ib1 = int(max(ir[ip] - 1, 0))
                    ib1 = int(min(ir[ip] + 1, nb-1))

                xl, yl, zl, al = curvature_radius(ib1, ib2, rmin, xl, yl, zl, nb,
                                        al, ip, sign_dir, s_loop, residual,
                                        nx, ny, nlen, idir, step, ir, xx_curv, yy_curv)                

                iz1 = max((ip + 1 - ngap), 0)
                if np.max(zl[iz1:ip+2]) <= 0:
                    break  # goto endsegm

            # ENDSEGM

            # RE-ORDERING LOOP COORDINATES
            
            if idir == 0:
                xloop = None
                yloop = None
                zloop = None
            xloop, yloop, zloop = reordering_loop(xl, yl, zl, ip, idir, xloop, yloop, zloop)

        ind = np.logical_and(xloop != 0, yloop != 0)
        nind = np.sum(ind)
        looplen = 0
        if nind > 1:
            # skip_struct
            xloop = xloop[ind]
            yloop = yloop[ind]
            zloop = zloop[ind]

            if iloop >= nloopmax:
                break  # end_trace

            np1 = len(xloop)
            s = np.zeros((np1))
            looplen = 0
            if np1 >= 2:
                for ip in range(1, np1):
                    s[ip] = s[ip - 1] + np.sqrt((xloop[ip] - xloop[ip - 1]) ** 2 + (yloop[ip] - yloop[ip - 1]) ** 2)
            looplen = s[np1-1]
            ns = max(int(looplen), 3)
            ss = np.arange(ns)

        # SKIP STRUCT
        if (looplen >= lmin):
            if iloop == 0:
                loopfile = None
            loopfile, iloop, loop_len, iloop_nstruc = loop_add(ns, reso, s, xloop, yloop, zloop, iloop, iloop_nstruc, istruc, loop_len, looplen, loopfile)

        # TEST DISPLAY

        # ERASE LOOP IN RESIDUAL IMAGE
        residual = erase_loop_in_residual(istart, jstart, wid, nx, ny, residual, xloop, yloop)

    # END_TRACE
    fluxmin = np.min(image1)
    fluxmax = np.max(image1)
    
    return loopfile, image2
