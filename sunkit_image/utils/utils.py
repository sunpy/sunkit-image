"""
This module contains a collection of functions of general utility.
"""

import warnings

import numpy as np
from scipy.interpolate import RectBivariateSpline
from skimage import measure

import astropy.units as u

import sunpy
from sunpy.map import all_coordinates_from_map

__all__ = [
    "apply_upsilon",
    "bin_edge_summary",
    "blackout_pixels_above_radius",
    "calculate_gamma",
    "equally_spaced_bins",
    "find_pixel_radii",
    "find_radial_bin_edges",
    "get_radial_intensity_summary",
    "points_in_poly",
    "reform2d",
    "remove_duplicate"
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
        Number of bins.

    Returns
    -------
    `numpy.ndarray`
        An array of shape ``(2, nbins)`` containing the bin edges.
    """
    if inner_value >= outer_value:
        msg = "The inner value must be strictly less than the outer value."
        raise ValueError(msg)
    if nbins <= 0:
        msg = "The number of bins must be strictly greater than 0."
        raise ValueError(msg)
    bin_edges = np.zeros((2, nbins))
    bin_edges[0, :] = np.arange(0, nbins)
    bin_edges[1, :] = np.arange(1, nbins + 1)
    return inner_value + bin_edges * (outer_value - inner_value) / nbins


def bin_edge_summary(r, binfit):
    """
    Return a summary of the bin edges.

    Parameters
    ----------
    r : `numpy.ndarray`
        An array of bin edges of shape (2, nbins) where nbins is the number of
        bins.
    binfit : {'center', 'left', 'right'}
        How to summarize the bin edges.

    Returns
    -------
    `numpy.ndarray`
        A one dimensional array of values that summarize the location of the bins.
    """
    if r.ndim != 2:
        msg = "The bin edges must be two-dimensional with shape (2, nbins)."
        raise ValueError(msg)
    if r.shape[0] != 2:
        msg = "The bin edges must be two-dimensional with shape (2, nbins)."
        raise ValueError(msg)
    if binfit == "center":
        summary = 0.5 * (r[0, :] + r[1, :])
    elif binfit == "left":
        summary = r[0, :]
    elif binfit == "right":
        summary = r[1, :]
    else:
        msg = 'Keyword "binfit" must have value "center", "left" or "right"'
        raise ValueError(msg)
    return summary


def find_pixel_radii(smap, scale=None):
    """
    Find the distance of every pixel in a map from the center of the Sun. The
    answer is returned in units of solar radii.

    Parameters
    ----------
    smap :`sunpy.map.Map`
        A SunPy map.
    scale : {`None` , `astropy.units.Quantity`}, optional
        The radius of the Sun expressed in map units. For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds. If None then the map is queried for the scale.

    Returns
    -------
    radii : `astropy.units.Quantity`
        An array the same shape as the input map. Each entry in the array
        gives the distance in solar radii of the pixel in the corresponding
        entry in the input map data.
    """
    # Calculate the coordinates of every pixel.
    coords = all_coordinates_from_map(smap)
    # TODO: check that the returned coordinates are indeed helioprojective cartesian
    # Calculate the radii of every pixel in helioprojective Cartesian
    # coordinate distance units.
    radii = np.sqrt(coords.Tx**2 + coords.Ty**2)
    # Re-scale the output to solar radii
    if scale is None:
        return u.R_sun * (radii / smap.rsun_obs)
    return u.R_sun * (radii / scale)


def get_radial_intensity_summary(smap, radial_bin_edges, scale=None, summary=np.mean, **summary_kwargs):
    """
    Get a summary statistic of the intensity in a map as a function of radius.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy map.
    radial_bin_edges : `astropy.units.Quantity`
        A two-dimensional array of bin edges of shape ``(2, nbins)`` where "nbins" is
        the number of bins.
    scale : { `None`, `astropy.units.Quantity` }, optional
        A length scale against which radial distances are measured, expressed
        in the map spatial units. For example, in AIA helioprojective
        Cartesian maps a useful length scale is the solar radius and is
        expressed in units of arcseconds.
    summary : ``function``, optional
        A function that returns a summary statistic of the distribution of intensity,
        at a given radius, for example `numpy.std`.
    summary_kwargs :`dict`, optional
        Keywords applicable to the summary function.

    Returns
    -------
    intensity summary : `numpy.ndarray`
        A summary statistic of the radial intensity in the bins defined by the
        bin edges.
    """
    s = smap.rsun_obs if scale is None else scale
    # Get the radial distance of every pixel from the center of the Sun.
    map_r = find_pixel_radii(smap, scale=s).to(u.R_sun)
    # Number of radial bins
    nbins = radial_bin_edges.shape[1]
    # Upper and lower edges
    lower_edge = [map_r > radial_bin_edges[0, i].to(u.R_sun) for i in range(nbins)]
    upper_edge = [map_r < radial_bin_edges[1, i].to(u.R_sun) for i in range(nbins)]
    # Calculate the summary statistic in the radial bins.
    with warnings.catch_warnings():
        # We want to ignore RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.asarray(
            [summary(smap.data[lower_edge[i] * upper_edge[i]], **summary_kwargs) for i in range(nbins)],
        )


def reform2d(array, factor=1):
    """
    Reform a 2d array by a given factor.

    Parameters
    ----------
    array : `numpy.ndarray`
        2d array to be reformed.
    factor : `int`, optional
        The array is going to be magnified by the factor. Default is 1.

    Returns
    -------
    `numpy.ndarray`
        Reformed array.
    """
    if not isinstance(factor, int):
        msg = "Parameter 'factor' must be an integer!"
        raise TypeError(msg)
    if len(np.shape(array)) != 2:
        msg = "Input array must be 2d!"
        raise ValueError(msg)
    if factor > 1:
        congridx = RectBivariateSpline(
            np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), array, kx=1, ky=1
        )
        return congridx(np.arange(0, array.shape[0], 1 / factor), np.arange(0, array.shape[1], 1 / factor))
    return array


def points_in_poly(poly):
    """
    Return polygon as grid of points inside polygon. Only works for polygons
    defined with points which are all integers.

    Parameters
    ----------
    poly : `numpy.ndarray`
        N x 2 array which defines all points at the edge of a polygon.

    Returns
    -------
    `numpy.ndarray`
        N x 2 array, all points within the polygon.
    """
    if np.shape(poly)[1] != 2:
        msg = "Polygon must be defined as a n x 2 array!"
        raise ValueError(msg)
    # Convert to integers
    poly = np.array(poly, dtype=int).tolist()
    xs, ys = zip(*poly, strict=True)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    # New polygon with the staring point as [0, 0]
    newPoly = [(int(x - minx), int(y - miny)) for (x, y) in poly]
    mask = measure.grid_points_in_poly((round(maxx - minx) + 1, round(maxy - miny) + 1), newPoly)
    # All points in polygon
    points = [[x + minx, y + miny] for x, y in zip(*np.nonzero(mask), strict=True)]
    # Add edge points if missing
    for p in poly:
        if p not in points:
            points.append(p)
    return points


def remove_duplicate(edge):
    """
    Remove duplicated points in a the edge of a polygon.

    Parameters
    ----------
    edge : `numpy.ndarray`
        N x 2 array which defines all points at the edge of a polygon.

    Returns
    -------
    `numpy.ndarray`
        Same as edge, but with duplicated points removed.
    """
    shape = np.shape(edge)
    if shape[1] != 2:
        msg = "Polygon must be defined as a n x 2 array!"
        raise ValueError(msg)
    new_edge = []
    for i in range(shape[0]):
        p = edge[i]
        if not isinstance(p, list):
            p = p.tolist()
        if p not in new_edge:
            new_edge.append(p)
    return new_edge


def _cross2d(x, y):
    # Dimension-2 input arrays were deprecated in 2.0.0.
    # See https://numpy.org/doc/stable/reference/generated/numpy.cross.html
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]


def calculate_gamma(pm, vel, pnorm, n):
    """
    Calculate gamma values.

    Parameters
    ----------
    pm : `numpy.ndarray`
        Vector from point "p" to point "m".
    vel : `numpy.ndarray`
        Velocity vector.
    pnorm : `numpy.ndarray`
        Mode of ``pm``.
    n : `int`
        Number of points.

    Returns
    -------
    `numpy.ndarray`
        calculated gamma values for velocity vector vel

    References
    ----------
    * Equation (8) in Laurent Graftieaux, Marc Michard and Nathalie Grosjean.
      Combining PIV, POD and vortex identification algorithms for the
      study of unsteady turbulent swirling flows.
      Meas. Sci. Technol. 12, 1422, 2001.
      (https://doi.org/10.1088/0957-0233/12/9/307)
    * Equation (1) in Jiajia Liu, Chris Nelson, Robert Erdelyi.
      Automated Swirl Detection Algorithm (ASDA) and Its Application to
      Simulation and Observational Data.
      Astrophys. J., 872, 22, 2019.
      (https://doi.org/10.3847/1538-4357/aabd34)
    """
    cross = _cross2d(pm, vel)
    vel_norm = np.linalg.norm(vel, axis=2)
    sint = cross / (pnorm * vel_norm + 1e-10)
    return np.nansum(sint, axis=1) / n


def apply_upsilon(data, upsilon=(0.5, 0.5)):
    """
    Apply the upsilon function to the input array.

    This function applies the upsilon function, a double-sided gamma adjustment,
    to the input array. It uses the specified exponents for the lower and upper halves
    of the array to normalize and stretch the values.

    Parameters
    ----------
    data : array-like
        Input array to be normalized and stretched.
    upsilon : float or tuple of float or None, optional
        Parameters for the upsilon function. Default is (0.5, 0.5). If None or contains all None, the original data is returned.
        If a single float is provided, both alpha and alpha_high are set to this value.

    Returns
    -------
    np.ndarray
        Normalized and stretched array.

    Raises
    ------
    TypeError
        If the input is a `sunpy.map.Map` object.
    """

    if upsilon is None:
        return data

    if isinstance(data, sunpy.map.GenericMap):
        msg = "Input data must be a raw ndarray, not a SunPy map object"
        raise TypeError(msg)

    if isinstance(upsilon, float):
        alpha = alpha_high = upsilon
    else:
        alpha, alpha_high = upsilon
        if alpha_high is None:
            alpha_high = 1.0
        elif alpha is None:
            alpha = 1.0

    in_array = np.asarray(data)

    # Calculate indices for low and high values
    mid = np.nanmean(in_array)
    lows = in_array < mid
    highs = in_array >= mid

    # Compute curve values
    curve_low = ((2 * in_array[lows]) ** alpha) / 2
    curve_high = -(((2 - 2 * in_array[highs]) ** alpha_high) / 2 - 1)

    # Create output array and assign calculated values
    out_curve = np.zeros_like(in_array)
    out_curve[lows] = curve_low
    out_curve[highs] = curve_high

    return out_curve


def blackout_pixels_above_radius(smap, radius_limit=1.5 * u.R_sun, fill=np.nan):
    """
    Black out any pixels above a certain radius in a SunPy map.

    Parameters
    ----------
    sunpy_map : `sunpy.map.GenericMap`
        The input sunpy map.
    radius_limit : `astropy.units.Quantity`
        The radius limit above which to black out pixels.
    fill : ``Any``, optional
        The value to use above the ``radius_limit``.
        Defaults to Nan.

    Returns
    -------
    `sunpy.map.GenericMap`
        A new sunpy map with pixels above the specified radius blacked out.
    """
    # Create a grid of coordinates corresponding to each pixel in the map
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # Create a mask for pixels above the radius limit
    mask = map_r > radius_limit

    # Apply the mask to the map data
    masked_data = np.where(mask, fill, smap.data)

    # Create a new map with the masked data
    return sunpy.map.Map(masked_data, smap.meta)


def find_radial_bin_edges(smap, radial_bin_edges=None):
    """
    Calculate radial bin edges for a solar map, either using provided edges or
    generating them automatically.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A sunpy Map containing the data to be binned.
    radial_bin_edges : `astropy.units.Quantity`, optional
        Pre-defined bin edges for radial binning. Should be a Quantity array with units
        of solar radii (u.R_sun) or pixels. If `None` (the default), bin edges
        will be automatically generated based on the map dimensions.

    Returns
    -------
    `astropy.units.Quantity`
        The final bin edges used for radial binning.
    `astropy.units.Quantity`
        Array of radial distances for each pixel in the map, matching the input
        map dimensions.
    """
    # Get the radii for every pixel, ensuring units are correct (in terms of pixels or solar radii)
    map_r = find_pixel_radii(smap)

    # Automatically generate radial bin edges if none are provided
    if radial_bin_edges is None:
        radial_bin_edges = equally_spaced_bins(0, np.max(map_r.value), smap.data.shape[0] // 2) * u.R_sun

    # Ensure radial_bin_edges are within the bounds of the map_r values
    if radial_bin_edges[1, -1] < np.max(map_r):
        radial_bin_edges = (
            equally_spaced_bins(
                inner_value=radial_bin_edges[0, 0].to(u.R_sun).value,
                outer_value=np.max(map_r.to(u.R_sun)).value,
                nbins=radial_bin_edges.shape[1] // 2,
            )
            * u.R_sun
        )
    return radial_bin_edges, map_r
