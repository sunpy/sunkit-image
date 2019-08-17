"""
This module contains a collection of functions of general utility.
"""
import numpy as np
import scipy.ndimage as ndimage
from scipy import interpolate

import astropy.units as u
from sunpy.coordinates import frames

__all__ = [
    "equally_spaced_bins",
    "bin_edge_summary",
    "find_pixel_radii",
    "bandpass_filter",
    "smooth",
    "erase_loop_in_residual",
    "curvature_radius",
    "initial_direction_finding",
    "loop_add",
]


def equally_spaced_bins(inner_value=1, outer_value=2, nbins=100):
    """
    Define a set of equally spaced bins between the specified inner and outer
    values. The inner value must be strictly less than the outer value.

    Parameters
    ----------
    inner_value : `float`
        The inner value of the bins.

    outer_value : `float`
        The outer value of the bins.

    nbins : `int`
        Number of bins

    Returns
    -------
    An array of shape (2, nbins) containing the bin edges.
    """
    if inner_value >= outer_value:
        raise ValueError("The inner value must be strictly less than the outer value.")

    if nbins <= 0:
        raise ValueError("The number of bins must be strictly greater than 0.")

    bin_edges = np.zeros((2, nbins))
    bin_edges[0, :] = np.arange(0, nbins)
    bin_edges[1, :] = np.arange(1, nbins + 1)
    return inner_value + bin_edges * (outer_value - inner_value) / nbins


def bin_edge_summary(r, binfit):
    """
    Return a summary of the bin edges.

    Parameters
    ----------
    r : `numpy.ndarray` like
        An array of bin edges of shape (2, nbins) where nbins is the number of
        bins.

    binfit : 'center' | 'left' | 'right'
        How to summarize the bin edges.

    Returns
    -------
    A one dimensional array of values that summarize the location of the bins.
    """
    if r.ndim != 2:
        raise ValueError("The bin edges must be two-dimensional with shape (2, nbins).")
    if r.shape[0] != 2:
        raise ValueError("The bin edges must be two-dimensional with shape (2, nbins).")

    if binfit == "center":
        summary = 0.5 * (r[0, :] + r[1, :])
    elif binfit == "left":
        summary = r[0, :]
    elif binfit == "right":
        summary = r[1, :]
    else:
        raise ValueError('Keyword "binfit" must have value "center", "left" or "right"')
    return summary


def find_pixel_radii(smap, scale=None):
    """
    Find the distance of every pixel in a map from the center of the Sun. The
    answer is returned in units of solar radii.

    Parameters
    ----------
    smap :
        A sunpy map object.

    scale : None | `~astropy.units.Quantity`
        The radius of the Sun expressed in map units. For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds. If None then the map is queried for the scale.

    Returns
    -------
    radii : `~astropy.units.Quantity`
        An array the same shape as the input map. Each entry in the array
        gives the distance in solar radii of the pixel in the corresponding
        entry in the input map data.
    """
    # Calculate all the x and y coordinates of every pixel in the map.
    x, y = np.meshgrid(*[np.arange(v.value) for v in smap.dimensions]) * u.pix

    # Calculate the helioprojective Cartesian co-ordinates of every pixel.
    coords = smap.pixel_to_world(x, y).transform_to(frames.Helioprojective)

    # Calculate the radii of every pixel in helioprojective Cartesian
    # co-ordinate distance units.
    radii = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2)

    # Re-scale the output to solar radii
    if scale is None:
        return u.R_sun * (radii / smap.rsun_obs)
    else:
        return u.R_sun * (radii / scale)


def get_radial_intensity_summary(
    smap, radial_bin_edges, scale=None, summary=np.mean, **summary_kwargs
):
    """
    Get a summary statistic of the intensity in a map as a function of radius.

    Parameters
    ----------
    smap : sunpy.map.Map
        A sunpy map.

    radial_bin_edges : `~astropy.units.Quantity`
        A two-dimensional array of bin edges of shape (2, nbins) where nbins is
        the number of bins.

    scale : None, `~astropy.units.Quantity`
        A length scale against which radial distances are measured, expressed
        in the map spatial units. For example, in AIA helioprojective
        Cartesian maps a useful length scale is the solar radius and is
        expressed in units of arcseconds.

    summary : `~function`
        A function that returns a summary statistic of the distribution of intensity,
        at a given radius, for example `~numpy.std`.

    summary_kwargs :`~dict`
        Keywords applicable to the summary function.

    Returns
    -------
    intensity summary : `~numpy.array`
        A summary statistic of the radial intensity in the bins defined by the
        bin edges.
    """
    if scale is None:
        s = smap.rsun_obs
    else:
        s = scale

    # Get the radial distance of every pixel from the center of the Sun.
    map_r = find_pixel_radii(smap, scale=s).to(u.R_sun)

    # Number of radial bins
    nbins = radial_bin_edges.shape[1]

    # Upper and lower edges
    lower_edge = [map_r > radial_bin_edges[0, i].to(u.R_sun) for i in range(0, nbins)]
    upper_edge = [map_r < radial_bin_edges[1, i].to(u.R_sun) for i in range(0, nbins)]

    # Calculate the summary statistic in the radial bins.
    return np.asarray(
        [
            summary(smap.data[lower_edge[i] * upper_edge[i]], **summary_kwargs)
            for i in range(0, nbins)
        ]
    )


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
    Python implementation of the IDL `smooth <https://www.harrisgeospatial.com/docs/smooth.html>`__.

    Parameters
    ----------
    image : `numpy.ndarray`
        Image to be filtered.
    width : `int`
        Width of the boxcar. The `width` should always be odd but if even value is given `width + 1` as the width of the boxcar.
    nanopt : {"propagate" | "replace"}
        It decides whether to `propagate` NAN's or `replace` them.

    Returns
    -------
    `numpy.ndarray`
        Smoothed image.
    """

    # make a copy of the array for the output:
    filtered=np.copy(image)

    # If width is even, add one
    if width % 2 == 0:
        width = width + 1

    # get the size of each dim of the input:
    r,c = image.shape

    # Assume that width, the width of the window is always square.
    startrc = int((width - 1)/2)
    stopr = int(r - ((width + 1)/2) + 1)
    stopc = int(c - ((width + 1)/2) + 1)

    # For all pixels within the border defined by the box size, calculate the average in the window.
    # There are two options:
    # Ignore NaNs and replace the value where possible.
    # Propagate the NaNs

    for col in range(startrc,stopc):
        # Calculate the window start and stop columns
        startwc = col - int(width/2) 
        stopwc = col + int(width/2) + 1
        for row in range (startrc,stopr):
            # Calculate the window start and stop rows
            startwr = row - int(width/2)
            stopwr = row + int(width/2) + 1
            # Extract the window
            window = image[startwr:stopwr, startwc:stopwc]
            if nanopt == 'replace':
                # If we're replacing Nans, then select only the finite elements
                window = window[np.isfinite(window)]
            # Calculate the mean of the window
            filtered[row,col] = np.mean(window)

    return filtered.astype(np.float32)


def erase_loop_in_residual(residual, istart, jstart, width, xloop, yloop):
    """
    Makes all the points in a loop and its vicinity as zero in the original image to prevent them from being
    traced again.

    Parameters
    ----------
    residual : `numpy.ndarray`
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

    nx, ny = residual.shape
    i3 = max(istart - width, 0)
    i4 = min(istart + width, nx - 1)
    j3 = max(jstart - width, 0)
    j4 = min(jstart + width, ny - 1)
    residual[i3:i4 + 1, j3:j4 + 1] = 0.

    nn = len(xloop)
    for iss in range(0, nn):
        i0 = min(max(int(xloop[iss]), 0), nx-1)
        i3 = max(int(i0 - width), 0)
        i4 = min(int(i0 + width), nx - 1)
        j0 = min(max(int(yloop[iss]), 0), ny-1)
        j3 = max(int(j0 - width), 0)
        j4 = min(int(j0 + width), ny - 1)
        residual[i3:i4 + 1, j3:j4 + 1] = 0.
    
    return residual


def loop_add(lengths, xloop, yloop, zloop, iloop, loops, loopfile):
    """
    Adds the current loop to the output structures by interpolating the coordinates

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
    loopfile : `numpy.ndarray`
        An array which stores data of the previous loops in the format as returned
        by the IDL function.

    Returns
    -------
    `tuple`
        It contains three elements: the first one is the updated `loopfile`, the second
        one is the updated `loops` list and the third one is the current loop number.
    """
    reso = 1
    ns = max(int(lengths[-1]), 3)
    nn = int(ns / reso + 0.5)

    ii = np.arange(nn) * reso
    interfunc = interpolate.interp1d(lengths, xloop, fill_value="extrapolate")
    xx = interfunc(ii)
    interfunc = interpolate.interp1d(lengths, yloop, fill_value="extrapolate")
    yy = interfunc(ii)
    interfunc = interpolate.interp1d(lengths, zloop, fill_value="extrapolate")
    ff = interfunc(ii)

    loopnum = np.ones((nn)) * iloop
    loop = np.c_[loopnum, xx, yy, ff, ii]

    if iloop == 0:
        loopfile = loop
    if iloop >= 1:
        loopfile = np.r_[loopfile, loop]
    iloop += 1

    current = []
    for i in range(0, len(xx)):
        current.append([xx[i], yy[i]])
    
    loops.append(current)

    return loopfile, loops, iloop


def initial_direction_finding(residual, xstart, ystart, nlen):
    """
    Finds the initial angle of the loop at the starting point.

    Parameters
    ----------
    residual : `numpy.ndarray`
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

    s0_loop = step * (np.arange(nlen, dtype=np.float32) - nlen // 2)
    alpha = np.pi * np.arange(na, dtype=np.float32) / np.float32(na)

    nx, ny = residual.shape
    flux_max = 0.
    for ia in range(0, na):
        x_ = xstart + s0_loop * np.float32(np.cos(alpha[ia]))
        y_ = ystart + s0_loop * np.float32(np.sin(alpha[ia]))
        ix = np.int_(x_ + 0.5)
        iy = np.int_(y_ + 0.5)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        flux_ = residual[ix, iy]
        flux = np.sum(np.maximum(flux_, 0.)) / np.float32(nlen)
        if flux > flux_max:
            flux_max = flux
            al = alpha[ia]

    return al


def curvature_radius(residual, rmin, xl, yl, zl, al, ir, ip, nlen, idir):
    """
    Finds the radius of curvature at the given loop point and then uses it to find the next point in the loop.

    Parameters
    ----------
    residual : `numpy.ndarray`
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

    nb = 30
    step = 1
    nx, ny = residual.shape

    s_loop = step * np.arange(nlen, dtype=np.float32)

    # This denotes loop tracing in forward direction
    if idir == 0:
        sign_dir = +1
    
    # This denotes loop tracing in backward direction
    if idir == 1:
        sign_dir = -1

    # `ib1` and `ib2` decide the range of radius in which the next point is to be searched
    if ip == 0:
        ib1 = 0
        ib2 = nb-1
    if ip >= 1:
        ib1 = int(max(ir[ip] - 1, 0))
        ib2 = int(min(ir[ip] + 1, nb-1))

    beta0 = al[ip] + np.float32(np.pi / 2)

    # Finding the assumed centre of the loop
    xcen = xl[ip] + rmin * np.float32(np.cos(beta0))
    ycen = yl[ip] + rmin * np.float32(np.sin(beta0))

    # Finding the radius associated with the next point by finding the maximum flux
    # along various radius values
    flux_max = 0.
    for ib in range(ib1, ib2 + 1):

        rad_i = rmin / (-1. + 2. * np.float32(ib) / np.float32(nb - 1))
        xcen_i = xl[ip] + (xcen - xl[ip]) * (rad_i / rmin)
        ycen_i = yl[ip] + (ycen - yl[ip]) * (rad_i / rmin)
        beta_i = beta0 + sign_dir * s_loop / rad_i
        x_ = xcen_i - rad_i * np.float32(np.cos(beta_i))
        y_ = ycen_i - rad_i * np.float32(np.sin(beta_i))
        ix = np.int_(x_ + 0.5)
        iy = np.int_(y_ + 0.5)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        flux_ = residual[ix, iy]
        flux = np.sum(np.maximum(flux_, 0.)) / np.float32(nlen)

        if flux > flux_max:
            flux_max = flux
            al[ip + 1] = al[ip] + sign_dir * (step / rad_i)
            ir[ip+1] = ib
            al_mid = (al[ip]+al[ip+1]) / 2.
            xl[ip+1] = xl[ip] + step * np.float32(np.cos(al_mid + np.pi * idir))
            yl[ip+1] = yl[ip] + step * np.float32(np.sin(al_mid + np.pi * idir))
            ix_ip = min(max(int(xl[ip + 1] + 0.5), 0), nx - 1)
            iy_ip = min(max(int(yl[ip + 1] + 0.5), 0), ny - 1)
            zl[ip + 1] = residual[ix_ip, iy_ip]
    
    return xl, yl, zl, al
