"""
This module contains functions that will enhance the trace out structures in an
image.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from sunkit_image.utils import bandpass_filter, erase_loop_in_residual, curvature_radius, initial_direction_finding, loop_add

__all__ = ["occult2"]


def occult2(image, nsm1, rmin, lmin, nstruc, nloop, ngap, qthresh1, qthresh2, file=False):
    """
    Implements the Oriented Coronal CUrved Loop Tracing (OCCULT-2) algorithm
    for loop tracing in solar images.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image on which loops are to be detected.
    nsm1 : `int`
        Low pass filter boxcar smoothing constant.
    rmin : `int`
        The minimum radius of curvature of the loop to be detected in pixels.
    lmin : `int`
        The length of the smallest loop to be detected in pixels.
    nstruc : `int`
        Maximum limit of traced structures.
    nloop : `int`
        Maximum number of detected loops.
    ngap : `int`
        Number of pixels in the loop below the flux threshold.
    qthresh1 : `float`
        The ratio of image base flux and median flux. All the pixels in the image below `qthresh1 * median` intensity value
        are made to zero before tracing the loops.
    qthresh2 : `float`
        The factor which determines noise in the image. All the intensity values between `qthresh2 * median` are considered
        to be noise. The median for noise is chosen after the base level is fixed.
    file : `bool`
        If set to `True` an IDL style output txt file is created with the name as `loop.txt`

    Returns
    -------
    `list`
        A list of all loop where each element is itself a list of points containg ``x`` and ``y`` coordinates for each point.

    References
    ----------
    * Markus J. Aschwanden, Bart De Pontieu, Eugene A. Katrukha.
      Optimization of Curvi-Linear Tracing Applied to Solar Physics and Biophysics.
      Entropy, vol. 15, issue 8, pp. 3007-3030
      https://doi.org/10.3390/e15083007
    """

    # Image is transposed because IDL works column major and python is row major. This is done
    # so that the python and the IDL codes look similar
    image = image.T

    # Defining all the other parameters as the IDL one.
    nloopmax = 10000
    npmax = 2000
    nsm2 = nsm1+2
    nlen = rmin
    
    wid = max(nsm2 // 2 - 1, 1)

    # BASE LEVEL: Removing the points below the base level
    zmed = np.median(image[image > 0])
    image = np.where(image > (zmed * qthresh1), image, zmed * qthresh1)

    # HIGHPASS FILTER
    image2 = bandpass_filter(image, nsm1, nsm2)
    nx, ny = image2.shape

    # ERASE BOUNDARIES ZONES (SMOOTHING EFFECTS)
    image2[:, 0:nsm2] = 0
    image2[:, ny - nsm2:] = 0
    image2[0:nsm2, :] = 0
    image2[nx - nsm2:, :] = 0

    # NOISE THRESHOLD
    zmed = np.median(image2[image2 > 0])
    thresh = zmed * qthresh2

    # Define the number of loops
    iloop = 0
    # The image with intensity less than zero removed
    residual = np.where(image2 > 0, image2, 0)

    for _ in range(0, nstruc):

        # Loop tracing begins at maximum flux position
        zstart = residual.max()

        # If maximum flux is less than noise threshold tracing stops
        if zstart <= thresh:  # goto: end_trace
            break

        # Points where the maximum flux is detected 
        max_coords = np.where(residual == zstart)
        istart, jstart = max_coords[0][0], max_coords[1][0]

        # TRACING LOOP STRUCTURE STEPWISE
        ip = 0
        ndir = 2
        for idir in range(0, ndir):

            # Creating arrays which will store all the loops points coordinates, flux, angle and radius
            xl = np.zeros((npmax + 1,))
            yl = np.zeros((npmax + 1,))
            zl = np.zeros((npmax + 1,))
            al = np.zeros((npmax + 1,))
            ir = np.zeros((npmax + 1,))

            # INITIAL DIRECTION FINDING
            xl[0] = istart
            yl[0] = jstart
            zl[0] = zstart
            
            # This will return the angle at the first point of the loop during every forward or backward pass
            al[0] = initial_direction_finding(residual, xl[0], yl[0], nlen)

            # `ip` denotes a point in the traced loop
            for ip in range(0, npmax):

                # The below function call will return the coordinate, flux and angle of the next point.
                xl, yl, zl, al = curvature_radius(residual, rmin, xl, yl, zl, al, ir, ip, nlen, idir)
                
                # This decides when to stop tracing the loop; when then last `ngap` pixels traced are below zero, the tracing will stop.
                iz1 = max((ip + 1 - ngap), 0)
                if np.max(zl[iz1:ip+2]) <= 0:
                    ip = max(iz1 - 1, 0)
                    break  # goto endsegm

            # ENDSEGM

            # RE-ORDERING LOOP COORDINATES
            # After the forward pass the loop points are flipped as the backward pass starts from the maximum flux point
            if idir == 0:
                xloop = np.flip(xl[0:ip+1])
                yloop = np.flip(yl[0:ip+1])
                zloop = np.flip(zl[0:ip+1])
                continue
            # After the backward pass the forward and backward traces are concatenated
            if idir == 1 and ip >= 1:
                xloop = np.concatenate([xloop, xl[1:ip+1]])
                yloop = np.concatenate([yloop, yl[1:ip+1]])
                zloop = np.concatenate([zloop, zl[1:ip+1]])
            else:
                break

        # Selecting only those loop points where both the coordinates are non-zero
        ind = np.logical_and(xloop != 0, yloop != 0)
        nind = np.sum(ind)
        looplen = 0
        if nind > 1:
            # skip_struct
            xloop = xloop[ind]
            yloop = yloop[ind]
            zloop = zloop[ind]

            # If number of traced loop is greater than maximum stop tracing
            if iloop >= nloopmax:
                break  # end_trace

            np1 = len(xloop)

            # Calculate the length of each loop
            s = np.zeros((np1), dtype=np.float32)
            looplen = 0
            if np1 >= 2:
                for ip in range(1, np1):
                    s[ip] = s[ip - 1] + np.sqrt((xloop[ip] - xloop[ip - 1]) ** 2 + (yloop[ip] - yloop[ip - 1]) ** 2)
            looplen = s[np1-1]

        # SKIP STRUCT: Only those loops are returned whose length is greater than the minimum specified
        if (looplen >= lmin):
            if iloop == 0:
                loopfile = None
                loops = []
            loopfile, loops, iloop = loop_add(s, xloop, yloop, zloop, iloop, loops, loopfile)

        # ERASE LOOP IN RESIDUAL IMAGE
        residual = erase_loop_in_residual(residual, istart, jstart, wid, xloop, yloop)

    if file is True:
        np.savetxt('loops.txt', loopfile, '%8.8f')
        
    del loopfile
    
    # END_TRACE     
    return loops
