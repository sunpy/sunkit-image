#
# This file contains a collection of functions of general utility
#

from __future__ import print_function, division

import numpy as np

import astropy.units as u

from sunpy.coordinates import frames

# Comparison operations that are allowed to be used
# when comparing the distance of a pixel in a map from
# the center of a map.
permitted_comparisons = (np.equal, np.not_equal, np.less, np.less_equal, np.greater, np.greater_equal)


def all_pixel_indices_from_map(smap):
    """
    Returns pixel pair indices of every pixel in a map.

    Parameters
    ----------
    smap : `~sunpy.map.Map`
        A SunPy map.

    Returns
    -------
    out : `~numpy.array`
        A numpy array with the all the pixel indices built from the
        dimensions of the map. The array is three dimensional.  The
        first dimension has size 2, the second and third dimensions
        are the same as the shape of the array holding the map image
        data.

    Example
    -------
    >>> import astropy.units as u
    >>> from sunpy.map import Map
    >>> from sunpy.data.sample import AIA_171_IMAGE
    >>> from sunkit_image.utils.utils import all_pixel_indices_from_map
    >>> smap = Map(AIA_171_IMAGE).submap((0, 0)*u.pix, (50, 60)*u.pix)
    >>> all_pixel_indices_from_map(smap).shape
    (2, 60, 50)

    """
    return np.meshgrid(*[np.arange(v.value) for v in smap.dimensions]) * u.pix


def all_coordinates_from_map(smap):
    """
    Returns the co-ordinates of every pixel in a map.

    Parameters
    ----------
    smap : `~sunpy.map.Map`
        A SunPy map.

    Returns
    -------
    out : `~astropy.coordinates.SkyCoord`
        An array of sky coordinates in the coordinate system of the input map.
        The array has the same shape as the same as the shape of the array
        holding the map image data.
    """
    x, y = all_pixel_indices_from_map(smap)
    return smap.pixel_to_world(x, y)


def find_pixel_radii(smap, scale=None):
    """
    Find the distance of every pixel in a map from the center of the Sun.
    The answer is returned in units of solar radii.

    Parameters
    ----------
    smap : `~sunpy.map.Map`
        A SunPy map.

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
    coords = all_coordinates_from_map(smap).transform_to(frames.Helioprojective)

    # Calculate the radii of every pixel in helioprojective Cartesian
    # co-ordinate distance units.
    radii = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2)

    # Re-scale the output to solar radii
    if scale is None:
        return u.R_sun * (radii / smap.rsun_obs)
    else:
        return u.R_sun * (radii / scale)


def locations_satisfying_condition_relative_to_radius(smap, comparison=np.less, scale=None, radius=None):
    """
    Return which locations in a map satisfy or fail the comparison of their distance from the
    center of the Sun with the input radius. Locations that satisfy the comparison are
    flagged as True. Locations that do not satisfy the comparison are flagged as False.

    Parameters
    ----------
    smap : `~sunpy.map.Map`
        A SunPy map.

    comparison : `~numpy.equal` | `~numpy.not_equal` | `~numpy.less` | `~numpy.less_equal` | `~numpy.greater` | `~numpy.greater_equal`
        The comparison operator applied.  The comparison is applied with the
        map locations on the left hand side of the inequality.

    scale : None | `~astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None then the map is queried for the scale.

    radius : None | `~astropy.units.Quantity`
        The radius used in the right hand side of the inequality. Must be
        in a unit convertible to solar radii.

    Returns
    -------
    locations : `~numpy.array`
        A numpy array of the same shape as the input map data with
        Boolean entries.  Locations that satisfy the comparison are
        flagged as True. Locations that do not satisfy the comparison
        are flagged as False.

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
        comparison_radius = radius.to(u.R_sun)

    # Find where the pixels are relative to the radius
    if comparison in permitted_comparisons:
        return comparison(map_pixel_radii, comparison_radius)
    else:
        names = ", ".join(["numpy.{:s}".format(n.__name__) for n in permitted_comparisons])
        raise ValueError('Comparison operator must be one of the following numpy function: {:s}'.format(names))


def locations_satisfying_radial_conditions(smap, comparison=(np.greater, np.less), scale=None, radii=None):
    """
    Find locations that satisfy a pair of radial distance comparisons
    simultaneously: such locations are flagged as True.  Other locations
    are flagged False.

    Parameters
    ----------
    smap : `~sunpy.map.Map`
        A SunPy map object.

    scale : None | `~astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None then the map is queried for the scale.

    comparison : `~tuple`
        The comparison operators applied.  Two comparisons are applied. Each
        comparison has the map locations on the left hand side of the inequality.
        The truth values of each comparison are combined using a logical AND
        operation. The permitted comparison operators are those permitted by
        `~sunkit_image.utils.utils.locations_satisfying_condition_relative_to_radius`.

    scale : None | `~astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None then the map is queried for the scale.

    radii : None | `~astropy.units.Quantity`
        The radii used in the right hand side of the inequality.

    Returns
    -------
    locations : `~numpy.array`
        A numpy array of the same shape as the input map data with
        Boolean entries.  Locations that satisfy the comparisons are
        flagged as True. Locations that do not satisfy the comparisons
        are flagged as False.
    """

    # Get the pixel scale
    if scale is None:
        map_scale = smap.rsun_obs
    else:
        map_scale = scale

    # Get the radii to calculate
    if radii is None:
        condition_radii = (1.0, 2.0) * u.R_sun
    elif len(radii) != 2:
        raise ValueError('The number of radii must be equal to 2.')
    else:
        condition_radii = radii

    # Get the number of comparison operators
    if len(comparison) != 2:
        raise ValueError('The number of comparison operators must be equal to 2.')

    condition1 = locations_satisfying_condition_relative_to_radius(smap,
                                                                   comparison=comparison[0],
                                                                   scale=map_scale,
                                                                   radius=condition_radii[0])
    condition2 = locations_satisfying_condition_relative_to_radius(smap,
                                                                   comparison=comparison[1],
                                                                   scale=None,
                                                                   radius=condition_radii[1])

    return np.logical_and(condition1, condition2)


def _equally_spaced_bins(inner_value=1, outer_value=2, nbins=100):
    """
    Define a set of equally spaced bins between the specified inner and outer
    values.  The inner value must be strictly less than the outer value.

    Parameters
    ----------
    inner_value : ``float`
        The inner value of the bins.

    outer_value : ``float`
        The outer value of the bins.

    nbins : ``int`
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
    summary : `~numpy.array`
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


def get_radial_intensity_summary(smap, radial_bin_edges,
                                 comparison=(np.greater, np.less),
                                 scale=None, summary=np.mean, **summary_kwargs):
    """
    Get a summary statistic of the intensity in a map as a function of radius.

    Parameters
    ----------
    smap : `~sunpy.map.Map`
        A SunPy map.

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

    # Number of radial bins
    nbins = radial_bin_edges.shape[1]

    # Find the pixels in all the radial bins
    annuli = list()
    for i in range(0, nbins):
        annuli.append(locations_satisfying_radial_conditions(smap,
                                                             comparison=comparison,
                                                             scale=scale,
                                                             radii=radial_bin_edges[:, i]))

    # Calculate the summary statistic in the radial bins.
    return np.asarray([summary(smap.data[annuli[i]], **summary_kwargs) for i in range(0, nbins)])
