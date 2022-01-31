"""
This module contains a collection of functions of general utility.
"""
import warnings

import numpy as np
from scipy.interpolate import interp2d
from skimage import measure

import astropy.units as u
from sunpy.map import all_coordinates_from_map

__all__ = [
    "bin_edge_summary",
    "calc_gamma",
    "equally_spaced_bins",
    "find_pixel_radii",
    "get_radial_intensity_summary",
    "points_in_poly",
    "reform2d",
    "remove_duplicate",
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
    r : `numpy.ndarray`
        An array of bin edges of shape (2, nbins) where nbins is the number of
        bins.
    binfit : {'center' | 'left' | 'right'}
        How to summarize the bin edges.

    Returns
    -------
    `numpy.ndarray`
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
    smap :`sunpy.map.Map`
        A SunPy map.
    scale : {`None` | `astropy.units.Quantity`}, optional
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
    # Calculate the co-ordinates of every pixel.
    coords = all_coordinates_from_map(smap)

    # TODO: check that the returned coordinates are indeed helioprojective cartesian

    # Calculate the radii of every pixel in helioprojective Cartesian
    # co-ordinate distance units.
    radii = np.sqrt(coords.Tx**2 + coords.Ty**2)

    # Re-scale the output to solar radii
    if scale is None:
        return u.R_sun * (radii / smap.rsun_obs)
    else:
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
    scale : {``None`` | `astropy.units.Quantity`}, optional
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
    with warnings.catch_warnings():
        # We want to ignore RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.asarray(
            [summary(smap.data[lower_edge[i] * upper_edge[i]], **summary_kwargs) for i in range(0, nbins)]
        )


def reform2d(array, factor=1):
    """
    Reform a 2d array by a given factor.

    Parameters
    ----------
    array : `numpy.ndarray`
        2d array to be reformed/
    factor : `int`, optional
        The array is going to be magnified by the factor. Default is 1.

    Returns
    -------
    `numpy.ndarray`
        Reformed array.
    """
    if not isinstance(factor, int):
        raise ValueError("Parameter 'factor' must be an integer!")

    if len(np.shape(array)) != 2:
        raise ValueError("Input array must be 2d!")

    if factor > 1:
        congridx = interp2d(np.arange(0, array.shape[0]), np.arange(0, array.shape[1]), array.T)
        array = congridx(np.arange(0, array.shape[0], 1 / factor), np.arange(0, array.shape[1], 1 / factor)).T

    return array


def points_in_poly(poly):
    """
    Return polygon as grid of points inside polygon. Only works for polygons
    defined with points which are all integers.

    Parameters
    ----------
    poly : `list` or `numpy.ndarray`
        N x 2 list which defines all points at the edge of a polygon.

    Returns
    -------
    `list`
        N x 2 array, all points within the polygon.
    """
    if np.shape(poly)[1] != 2:
        raise ValueError("Polygon must be defined as a n x 2 array!")

    # convert to integers
    poly = np.array(poly, dtype=int).tolist()

    xs, ys = zip(*poly)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    # New polygon with the staring point as [0, 0]
    newPoly = [(int(x - minx), int(y - miny)) for (x, y) in poly]
    mask = measure.grid_points_in_poly((round(maxx - minx) + 1, round(maxy - miny) + 1), newPoly)
    # all points in polygon
    points = [[x + minx, y + miny] for x, y in zip(*np.nonzero(mask))]

    # add edge points if missing
    for p in poly:
        if p not in points:
            points.append(p)

    return points


def remove_duplicate(edge):
    """
    Remove duplicated points in a the edge of a polygon.

    Parameters
    ----------
    edge : `list` or `numpy.ndarray`
        N x 2 list which defines all points at the edge of a polygon.

    Returns
    -------
    `list`
        Same as edge, but with duplicated points removed.
    """

    shape = np.shape(edge)
    if shape[1] != 2:
        raise ValueError("Polygon must be defined as a n x 2 array!")

    new_edge = []
    for i in range(shape[0]):
        p = edge[i]
        if not isinstance(p, list):
            p = p.tolist()
        if p not in new_edge:
            new_edge.append(p)

    return new_edge


def calc_gamma(pm, vel, pnorm, N):
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
    N : `int`
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

    cross = np.cross(pm, vel)
    vel_norm = np.linalg.norm(vel, axis=2)
    sint = cross / (pnorm * vel_norm + 1e-10)

    return np.nansum(sint, axis=1) / N
