"""
This module contains functions that will the trace out coronal loop-like
structures in an image.
"""

import numpy as np
from scipy import interpolate

__all__ = [
    "occult2",
    "bandpass_filter",
    "curvature_radius",
    "erase_loop_in_image",
    "initial_direction_finding",
    "loop_add",
    "smooth",
]


def occult2(image, nsm1, rmin, lmin, nstruc, ngap, qthresh1, qthresh2):
    """
    Implements the Oriented Coronal CUrved Loop Tracing (OCCULT-2) algorithm
    for loop tracing in images.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image in which loops are to be detected.
    nsm1 : `int`
        Low pass filter boxcar smoothing constant.
    rmin : `int`
        The minimum radius of curvature of the loop to be detected in pixels.
    lmin : `int`
        The length of the smallest loop to be detected in pixels.
    nstruc : `int`
        Maximum limit of traced structures.
    ngap : `int`
        Number of pixels in the loop below the flux threshold.
    qthresh1 : `float`
        The ratio of image base flux and median flux. All the pixels in the image below
        `qthresh1 * median` intensity value are made to zero before tracing the loops.
    qthresh2 : `float`
        The factor which determines noise in the image. All the intensity values between
        `qthresh2 * median` are considered to be noise. The median for noise is chosen
        after the base level is fixed.

    Returns
    -------
    `list`
        A list of all loop where each element is itself a list of points containing
        ``x`` and ``y`` coordinates for each point.

    References
    ----------
    * Markus J. Aschwanden, Bart De Pontieu, Eugene A. Katrukha.
      Optimization of Curvi-Linear Tracing Applied to Solar Physics and Biophysics.
      Entropy, vol. 15, issue 8, pp. 3007-3030
      https://doi.org/10.3390/e15083007
    """

    image = image.astype(np.float32)

    # Image is transposed because IDL works column major and python is row major. This is done
    # so that the python and the IDL codes look similar
    image = image.T

    # Defining all the other parameters as the IDL one.
    # The maximum number of loops that can be detected
    nloopmax = 10000

    # The maximum number of points in a loop
    npmax = 2000

    # High pass filter boxcar window size
    nsm2 = nsm1 + 2

    # The length of the tracing curved element
    nlen = rmin

    wid = max(nsm2 // 2 - 1, 1)

    # BASE LEVEL: Removing the points below the base level
    zmed = np.median(image[image > 0])
    image = np.where(image > (zmed * qthresh1), image, zmed * qthresh1)

    # BANDPASS FILTER
    image2 = bandpass_filter(image, nsm1, nsm2)
    nx, ny = image2.shape

    # ERASE BOUNDARIES ZONES (SMOOTHING EFFECTS)
    image2[:, 0:nsm2] = 0.0
    image2[:, ny - nsm2 :] = 0.0
    image2[0:nsm2, :] = 0.0
    image2[nx - nsm2 :, :] = 0.0

    if (not np.count_nonzero(image2)) is True:
        raise RuntimeError(
            "The filter size is very large compared to the size of the image."
            + " The entire image zeros out while smoothing the image edges after filtering."
        )

    # NOISE THRESHOLD
    zmed = np.median(image2[image2 > 0])
    thresh = zmed * qthresh2

    # Defines the current number of loop being traced
    iloop = 0

    # The image with intensity less than zero removed
    residual = np.where(image2 > 0, image2, 0)

    # Creating the structure in which the loops will be finally stored
    loops = []

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
        # The point number in the current loop being traced
        ip = 0

        # The two directions in bidirectional tracing of loops
        ndir = 2

        for idir in range(0, ndir):

            # Creating arrays which will store all the loops points coordinates, flux,
            # angle and radius.
            # xl, yl are the x and y coordinates
            xl = np.zeros((npmax + 1,), dtype=np.float32)
            yl = np.zeros((npmax + 1,), dtype=np.float32)

            # zl is the flux at each loop point
            zl = np.zeros((npmax + 1,), dtype=np.float32)

            # al, rl are the angles and radius involved with every loop point
            al = np.zeros((npmax + 1,), dtype=np.float32)
            ir = np.zeros((npmax + 1,), dtype=np.float32)

            # INITIAL DIRECTION FINDING
            xl[0] = istart
            yl[0] = jstart
            zl[0] = zstart

            # This will return the angle at the first point of the loop during every
            # forward or backward pass
            al[0] = initial_direction_finding(residual, xl[0], yl[0], nlen)

            # `ip` denotes a point in the traced loop
            for ip in range(0, npmax):

                # The below function call will return the coordinate, flux and angle
                # of the next point.
                xl, yl, zl, al = curvature_radius(residual, rmin, xl, yl, zl, al, ir, ip, nlen, idir)

                # This decides when to stop tracing the loop; when then last `ngap` pixels traced
                # are below zero, the tracing will stop.
                iz1 = max((ip + 1 - ngap), 0)
                if np.max(zl[iz1 : ip + 2]) <= 0:
                    ip = max(iz1 - 1, 0)
                    break  # goto endsegm

            # ENDSEGM

            # RE-ORDERING LOOP COORDINATES
            # After the forward pass the loop points are flipped as the backward pass starts
            # from the maximum flux point
            if idir == 0:
                xloop = np.flip(xl[0 : ip + 1])
                yloop = np.flip(yl[0 : ip + 1])
                zloop = np.flip(zl[0 : ip + 1])
                continue
            # After the backward pass the forward and backward traces are concatenated
            if idir == 1 and ip >= 1:
                xloop = np.concatenate([xloop, xl[1 : ip + 1]])
                yloop = np.concatenate([yloop, yl[1 : ip + 1]])
                zloop = np.concatenate([zloop, zl[1 : ip + 1]])
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
            looplen = s[np1 - 1]

        # SKIP STRUCT: Only those loops are returned whose length is greater than the minimum
        # specified
        if looplen >= lmin:
            loops, iloop = loop_add(s, xloop, yloop, zloop, iloop, loops)

        # ERASE LOOP IN RESIDUAL IMAGE
        residual = erase_loop_in_image(residual, istart, jstart, wid, xloop, yloop)

    # END_TRACE
    return loops


# The functions below this are subroutines for the OCCULT 2.
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
    `numpy.ndarray`
        Bandpass filtered image.
    """

    if nsm1 >= nsm2:
        raise ValueError("nsm1 should be less than nsm2")

    if nsm1 <= 2:
        return image - smooth(image, nsm2, "replace")

    if nsm1 >= 3:
        return smooth(image, nsm1, "replace") - smooth(image, nsm2, "replace")


def smooth(image, width, nanopt="replace"):
    """
    Python implementation of the IDL's ``smooth``.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image to be filtered.
    width : `int`
        Width of the boxcar window. The `width` should always be odd but if even value is given then
        `width + 1` is used as the width of the boxcar.
    nanopt : {"propagate" | "replace"}
        It decides whether to `propagate` NAN's or `replace` them.

    Returns
    -------
    `numpy.ndarray`
        Smoothed image.

    References
    ----------
    * https://www.harrisgeospatial.com/docs/smooth.html
    * Emmalg's answer on stackoverflow https://stackoverflow.com/a/35777966
    """

    # Make a copy of the array for the output:
    filtered = np.copy(image)

    # If width is even, add one
    if width % 2 == 0:
        width = width + 1

    # get the size of each dim of the input:
    r, c = image.shape

    # Assume that width, the width of the window is always square.
    startrc = int((width - 1) / 2)
    stopr = int(r - ((width + 1) / 2) + 1)
    stopc = int(c - ((width + 1) / 2) + 1)

    # For all pixels within the border defined by the box size, calculate the average in the window.
    # There are two options:
    # Ignore NaNs and replace the value where possible.
    # Propagate the NaNs

    for col in range(startrc, stopc):
        # Calculate the window start and stop columns
        startwc = col - int(width / 2)
        stopwc = col + int(width / 2) + 1
        for row in range(startrc, stopr):
            # Calculate the window start and stop rows
            startwr = row - int(width / 2)
            stopwr = row + int(width / 2) + 1
            # Extract the window
            window = image[startwr:stopwr, startwc:stopwc]
            if nanopt == "replace":
                # If we're replacing Nans, then select only the finite elements
                window = window[np.isfinite(window)]
            # Calculate the mean of the window
            filtered[row, col] = np.mean(window)

    return filtered.astype(np.float32)


def erase_loop_in_image(image, istart, jstart, width, xloop, yloop):
    """
    Makes all the points in a loop and its vicinity as zero in the original
    image to prevent them from being traced again.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image in which the points of a loop and surrounding it are to be made zero.
    istart : `int`
        The ``x`` coordinate of the starting point of the loop.
    jstart : `int`
        The ``y`` coordinate of the starting point of the loop.
    width : `int`
        The number of pixels around a loop point which are also to be removed.
    xloop : `numpy.ndarray`
        The ``x`` coordinates of all the loop points.
    yloop : `numpy.ndarray`
        The ``y`` coordinates of all the loop points.

    Returns
    -------
    `numpy.ndarray`
        Image with the loop and surrounding points zeroed out..
    """

    nx, ny = image.shape

    # The points surrounding the first point of the loop are zeroed out
    xstart = max(istart - width, 0)
    xend = min(istart + width, nx - 1)
    ystart = max(jstart - width, 0)
    yend = min(jstart + width, ny - 1)
    image[xstart : xend + 1, ystart : yend + 1] = 0.0

    # All the points surrounding the loops are zeroed out
    for point in range(0, len(xloop)):

        i0 = min(max(int(xloop[point]), 0), nx - 1)
        xstart = max(int(i0 - width), 0)
        xend = min(int(i0 + width), nx - 1)
        j0 = min(max(int(yloop[point]), 0), ny - 1)
        ystart = max(int(j0 - width), 0)
        yend = min(int(j0 + width), ny - 1)
        image[xstart : xend + 1, ystart : yend + 1] = 0.0

    return image


def loop_add(lengths, xloop, yloop, zloop, iloop, loops):
    """
    Adds the current loop to the output structures by interpolating the
    coordinates.

    Parameters
    ----------
    lengths : `numpy.ndarray`
        The length of loop at every point from the starting point.
    xloop : `numpy.ndarray`
        The ``x`` coordinates of all the points of the loop.
    yloop : `numpy.ndarray`
        The ``y`` coordinates of all the points of the loop.
    zloop : `numpy.ndarray`
        The flux intensity at every point of the loop.
    iloop : `int`
        The current loop number.
    loops : `list`
        It is a list of lists which contains all the previous loops.

    Returns
    -------
    `tuple`
        It contains three elements: the first one is the updated `loopfile`, the second
        one is the updated `loops` list and the third one is the current loop number.
    """

    # The resolution between the points
    reso = 1

    # The length of the loop must be greater than 3 to interpolate
    nlen = max(int(lengths[-1]), 3)

    # The number of points in the final loop
    num_points = int(nlen / reso + 0.5)

    # All the coordinates and the flux values are interpolated
    interp_points = np.arange(num_points) * reso

    # The one dimensional interpolation function created for interpolating x coordinates
    interfunc = interpolate.interp1d(lengths, xloop, fill_value="extrapolate")
    x_interp = interfunc(interp_points)

    # The one dimensional interpolation function created for interpolating y coordinates
    interfunc = interpolate.interp1d(lengths, yloop, fill_value="extrapolate")
    y_interp = interfunc(interp_points)

    iloop += 1

    # The current loop which will contain its points
    current = []
    for i in range(0, len(x_interp)):
        current.append([x_interp[i], y_interp[i]])

    loops.append(current)

    return loops, iloop


def initial_direction_finding(image, xstart, ystart, nlen):
    """
    Finds the initial angle of the loop at the starting point.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image in which the loops are being detected.
    xstart : `int`
        The ``x`` coordinates of the starting point of the loop.
    ystart : `int`
        The ``y`` coordinates of the starting point of the loop.
    nlen : `int`
        The length of the guiding segment.

    Returns
    -------
    `float`
        The angle of the starting point of the loop.
    """

    # The number of steps to be taken to move from one point to another
    step = 1
    na = 180

    # Shape of the input array
    nx, ny = image.shape

    # Creating the bidirectional tracing segment
    trace_seg_bi = step * (np.arange(nlen, dtype=np.float32) - nlen // 2).reshape((-1, 1))

    # Creating an array of all angles between 0 to 180 degree
    angles = np.pi * np.arange(na, dtype=np.float32) / np.float32(na).reshape((1, -1))

    # Calculating the possible x and y values when you move the tracing
    # segment along a particular angle
    x_pos = xstart + np.matmul(trace_seg_bi, np.float32(np.cos(angles)))
    y_pos = ystart + np.matmul(trace_seg_bi, np.float32(np.sin(angles)))

    # Taking the ceil as images can be indexed by pixels
    ix = (x_pos + 0.5).astype(int)
    iy = (y_pos + 0.5).astype(int)

    # All the coordinate values should be within the input range
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # Calculating the mean flux at possible x and y locations
    flux_ = image[ix, iy]
    flux = np.sum(np.maximum(flux_, 0.0), axis=0) / np.float32(nlen)

    # Returning the angle along which the flux is maximum
    return angles[0, np.argmax(flux)]


def curvature_radius(image, rmin, xl, yl, zl, al, ir, ip, nlen, idir):
    """
    Finds the radius of curvature at the given loop point and then uses it to
    find the next point in the loop.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image in which the loops are being detected.
    rmin : `float`
        The minimum radius of curvature of any point in the loop.
    xl : `numpy.ndarray`
        The ``x`` coordinates of all the points of the loop.
    yl : `nump.ndarray`
        The ``y`` coordinates of all the points of the loop.
    zl : `nump.ndarray`
        The flux intensity at all the points of the loop.
    al : `nump.ndarray`
        The angles associated with every point of the loop.
    ir : `nump.ndarray`
        The radius associated with every point of the loop.
    ip : `int`
        The current number of the point being traced in a loop.
    nlen : `int`
        The length of the guiding segment.
    idir : `int`
        The flag which denotes whether it is a forward pass or a backward pass.
        `0` denotes forward pass and `1` denotes backward pass.

    Returns
    -------
    `float`
        The angle of the starting point of the loop.
    """

    # Number of radial segments to be searched
    rad_segments = 30

    # The number of steps to be taken to move from one point to another
    step = 1
    nx, ny = image.shape

    # The unidirectional tracing segment
    trace_seg_uni = step * np.arange(nlen, dtype=np.float32).reshape((-1, 1))

    # This denotes loop tracing in forward direction
    if idir == 0:
        sign_dir = +1

    # This denotes loop tracing in backward direction
    if idir == 1:
        sign_dir = -1

    # `ib1` and `ib2` decide the range of radius in which the next point is to be searched
    if ip == 0:
        ib1 = 0
        ib2 = rad_segments - 1
    if ip >= 1:
        ib1 = int(max(ir[ip] - 1, 0))
        ib2 = int(min(ir[ip] + 1, rad_segments - 1))

    # See Eqn. 6 in the paper. Getting the values of all the valid radii
    rad_i = rmin / (-1.0 + 2.0 * np.arange(ib1, ib2 + 1, dtype=np.float32) / np.float32(rad_segments - 1)).reshape(
        (1, -1)
    )

    # See Eqn 16.
    beta0 = al[ip] + np.float32(np.pi / 2)

    # Finding the assumed centre of the loop
    # See Eqn 17, 18.
    xcen = xl[ip] + rmin * np.float32(np.cos(beta0))
    ycen = yl[ip] + rmin * np.float32(np.sin(beta0))

    # See Eqn 19, 20.
    xcen_i = xl[ip] + (xcen - xl[ip]) * (rad_i / rmin)
    ycen_i = yl[ip] + (ycen - yl[ip]) * (rad_i / rmin)

    # All the possible values of angle of the curved segment from cente
    # See Eqn 21.
    beta_i = beta0 + sign_dir * np.float32(np.matmul(trace_seg_uni, 1 / rad_i))

    # Getting the possible values of the coordinates
    x_pos = xcen_i - rad_i * np.float32(np.cos(beta_i))
    y_pos = ycen_i - rad_i * np.float32(np.sin(beta_i))

    # Taking the ceil as images can be indexed by pixels
    ix = (x_pos + 0.5).astype(int)
    iy = (y_pos + 0.5).astype(int)

    # All the coordinate values should be within the input range
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    # Calculating the mean flux at possible x and y locations
    flux_ = image[ix, iy]

    # Finding the average flux at every radii
    flux = np.sum(np.maximum(flux_, 0.0), axis=0) / np.float32(nlen)

    # Finding the maximum flux radii
    v = np.argmax(flux)

    # Getting the direction angle for the next point
    # See Eqn 25.
    al[ip + 1] = al[ip] + sign_dir * (step / rad_i[0, v])
    ir[ip + 1] = ib1 + v

    # See Eqn 26.
    al_mid = (al[ip] + al[ip + 1]) / 2.0

    # Coordinates of the next point in the loop
    xl[ip + 1] = xl[ip] + step * np.float32(np.cos(al_mid + np.pi * idir))
    yl[ip + 1] = yl[ip] + step * np.float32(np.sin(al_mid + np.pi * idir))

    # Bringing the coordinates values in the valid pixel range
    ix_ip = min(max(int(xl[ip + 1] + 0.5), 0), nx - 1)
    iy_ip = min(max(int(yl[ip + 1] + 0.5), 0), ny - 1)
    zl[ip + 1] = image[ix_ip, iy_ip]

    return xl, yl, zl, al
