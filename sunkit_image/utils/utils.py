"""
This module contains a collection of functions of general utility.
"""
import warnings

import numpy as np

import astropy.units as u
from sunpy.map import all_coordinates_from_map

__all__ = [
    "bin_edge_summary",
    "equally_spaced_bins",
    "find_pixel_radii",
    "get_radial_intensity_summary",
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
