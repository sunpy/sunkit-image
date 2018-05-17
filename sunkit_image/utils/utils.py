#
# This file contains a collection of functions of general utility
#

from __future__ import print_function, division

import numpy as np

import astropy.units as u

from sunpy.coordinates import frames


def all_pixels_from_map(smap):
    """
    Returns pixel pair indices of every pixel in a map.

    :param smap:
    :return:
    """
    return np.meshgrid(*[np.arange(v.value) for v in smap.dimensions]) * u.pix


def all_coordinates_from_map(smap, coordinate_system=frames.Helioprojective):
    """
    Returns the co-ordinates of every pixel in a map.

    :param smap:
    :param coordinate_system:
    :return:
    """
    x, y = all_pixels_from_map(smap)
    return smap.pixel_to_world(x, y).transform_to(coordinate_system)


def _equally_spaced_bins(inner_value=1, outer_value=2, nbins=100):
    """
    Define a set of equally spaced bins between the specified inner and outer
    values.  The inner value must be strictly less than the outer value.

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
        raise ValueError('The inner value must be strictly less than the outer value.')

    if nbins <= 0:
        raise ValueError('The number of bins must be strictly greater than 0.')

    bin_edges = np.zeros((2, nbins))
    bin_edges[0, :] = np.arange(0, nbins)
    bin_edges[1, :] = np.arange(1, nbins+1)
    return inner_value + bin_edges * (outer_value - inner_value) / nbins


def bin_edge_summary(r, binfit):
    """
    Return a summary of the bin edges.

    Parameters
    ----------
    r :  `numpy.ndarray` like
        An array of bin edges of shape (2, nbins) where nbins is the number of
        bins.

    binfit : 'center' | 'left' | 'right'
        How to summarize the bin edges.

    Returns
    -------
    A one dimensional array of values that summarize the location of the bins.

    """
    if r.ndim != 2:
        raise ValueError('The bin edges must be two-dimensional with shape (2, nbins).')
    if r.shape[0] != 2:
        raise ValueError('The bin edges must be two-dimensional with shape (2, nbins).')

    if binfit == 'center':
        summary = 0.5 * (r[0, :] + r[1, :])
    elif binfit == 'left':
        summary = r[0, :]
    elif binfit == 'right':
        summary = r[1, :]
    else:
        raise ValueError(
            'Keyword "binfit" must have value "center", "left" or "right"')
    return summary


def find_pixel_radii(smap, scale=None):
    """
    Find the distance of every pixel in a map from the center of the Sun.
    The answer is returned in units of solar radii.

    Parameters
    ----------
    smap :
        A sunpy map object.

    scale : None | `~astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None then the map is queried for the scale.

    Returns
    -------
    radii : `~astropy.units.Quantity`
        An array the same shape as the input map.  Each entry in the array
        gives the distance in solar radii of the pixel in the corresponding
        entry in the input map data.
    """

    # Calculate the helioprojective Cartesian co-ordinates of every pixel.
    coords = all_coordinates_from_map(smap)

    # Calculate the radii of every pixel in helioprojective Cartesian
    # co-ordinate distance units.
    radii = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2)

    # Re-scale the output to solar radii
    if scale is None:
        return u.R_sun * (radii / smap.rsun_obs)
    else:
        return u.R_sun * (radii / scale)


def map_coordinates_relative_to_radius(smap, comparison='<', scale=None, radius=None):
    """
    Return the co-ordinates that satisfy the comparison relative to the radius.

    Parameters
    ----------
    smap :
        A sunpy map object.

    comparison : `~str`
        A string as

    scale : None | `~astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None then the map is queried for the scale.


    radius : None | `~astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None then the map is queried for the scale.

    Returns
    -------
    radii : `~astropy.units.Quantity`
        An array the same shape as the input map.  Each entry in the array
        gives the distance in solar radii of the pixel in the corresponding
        entry in the input map data.
    """
    # Calculate the truth values of each of the map pixels using the comparison
    condition = map_pixels_condition_relative_to_radius(smap, comparison=comparison, scale=scale, radius=radius)

    # Return the coordinates of all the map pixels
    coords = all_coordinates_from_map(smap)

    # Return the co-ordinates that satisfy the condition
    return coords[condition]


def map_pixels_condition_relative_to_radius(smap, comparison='<', scale=None, radius=None):
    """

    :param smap:
    :param comparison:
    :param scale:
    :param radius:
    :return:
    """

    # Get the pixel scale
    if scale is None:
        map_scale = smap.rsun_obs
    else:
        map_scale = scale

    # Find which pixels
    map_pixel_radii = find_pixel_radii(smap, scale=map_scale)

    # Get the radius below which
    if radius is None:
        comparison_radius = 1.0 * u.R_sun
    else:
        comparison_radius = radius

    # Find where the pixels are relative to the radius
    if comparison == '<':
        locations = map_pixel_radii < comparison_radius
    elif comparison == '<=':
        locations = map_pixel_radii <= comparison_radius
    elif comparison == '>':
        locations = map_pixel_radii > comparison_radius
    elif comparison == '>=':
        locations = map_pixel_radii >= comparison_radius
    else:
        raise ValueError('Comparison operator not understood')

    # Return results
    return locations


def map_pixels_condition_annulus(smap, comparison=('>', '<'), scale=None, radii=None):
    """
    Find which pixels are in an annular region.

    :param smap:
    :param scale:
    :param radii:
    :return:
    """

    # Get the pixel scale
    if scale is None:
        map_scale = smap.rsun_obs
    else:
        map_scale = scale

    # Get the radius below which
    if radii is None:
        annulus_radii = (1.0, 2.0) * u.R_sun
    elif radii[0] >= radii[1]:
        raise ValueError('Inner radius of annulus must be strictly less than the outer radius of the annulus')
    else:
        annulus_radii = radii

    greater_than_inner_radius = map_pixels_condition_relative_to_radius(smap,
                                                                        comparison=comparison[0],
                                                                        scale=map_scale,
                                                                        radius=annulus_radii[0])
    less_than_outer_radius = map_pixels_condition_relative_to_radius(smap,
                                                                     comparison=comparison[1],
                                                                     scale=None,
                                                                     radius=annulus_radii[1])

    return np.logical_and(greater_than_inner_radius, less_than_outer_radius)

def get_radial_intensity_summary(smap, radial_bin_edges, scale=None, summary=np.mean, **summary_kwargs):
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
    return np.asarray([summary(smap.data[lower_edge[i] * upper_edge[i]], **summary_kwargs) for i in range(0, nbins)])
