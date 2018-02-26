#
# This file contains a collection of functions of general utility
#

from __future__ import print_function, division

import numpy as np

import astropy.units as u

from sunpy.coordinates import frames


def _equally_spaced_bins(inner_radius=1, outer_radius=2, nbins=100):
    """
    Define a set of equally spaced bins between the specified inner and outer
    radii.

    Parameters
    ----------
    inner_radius : `float`
        The inner radius of the bins.

    outer_radius : `float`
        The outer radius of the bins.

    nbins : `int`
        Number of bins

    Returns
    -------
    An array of size [2, nbins] containing the bin edges.
    """
    bin_edges = np.zeros((2, nbins))
    bin_edges[0, :] = np.arange(0, nbins)
    bin_edges[1, :] = np.arange(1, nbins+1)
    return bin_edges * (outer_radius - inner_radius) / nbins


def bin_edge_summary(r, binfit):
    """
    Return a summary of the bin edges.

    Parameters
    ----------
    r :  `numpy.ndarray` like
        An array of bin edges of size [2, nbins] where nbins is the number of
        bins.

    binfit : 'center' | 'left' | 'right'
        How to summarize the bin edges.

    Returns
    -------
    A one dimensional array of values that summarize the location of the bins.

    """
    if binfit == 'center':
        rfit = 0.5 * (r[0, :] + r[1, :])
    elif binfit == 'left':
        rfit = r[0, :]
    elif binfit == 'right':
        rfit = r[1, :]
    else:
        raise ValueError(
            'Keyword "binfit" must have value "center", "left" or "right"')
    return rfit


def find_pixel_radii(smap, scale=None):
    """
    Find the distance of every pixel in a map from the center of the Sun.
    The answer is returned in units of solar radii.

    Parameters
    ----------
    smap :
        A sunpy map object.

    scale :
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.

    Returns
    -------
    radii : `~astropy.units.Quantity`
        An array the same shape as the input map.  Each entry in the array
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


def get_radial_intensity_summary(smap, radial_bin_edges, scale=None, summary=np.mean, **summary_kwargs):
    """
    Get a summary statistic of the intensity in a map as a function of radius.

    Parameters
    ----------
    smap : sunpy.map.Map
        A sunpy map.

    radial_bin_edges : `~astropy.units.Quantity`
        A two-dimensional array of bin edges of size [2, nbins] where nbins is
        the number of bins.

    Keywords
    --------
    scale : None, `~astropy.units.Quantity`
        A length scale against which radial distances are measured, expressed
        in the map spatial units. For example, in AIA helioprojective
        Cartesian maps a useful length scale is the solar radius and is
        expressed in units of arcseconds.

    summary : `function`
        ???

    summary_kwargs :`dict`
        ???

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
    return np.asarray([summary(smap.data[lower_edge[i] * upper_edge[i]], **summary_kwargs) for i in range(0, nbins)])
