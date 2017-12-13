from __future__ import print_function, division

import numpy as np

import astropy.units as u

import sunpy.map

from sunpy.coordinates import frames


def _radial_bins(r, map_r):
    """

    Parameters
    ----------
    r
    map_r

    Returns
    -------

    """
    if r is None:
        return np.arange(1, np.max(map_r).to(u.R_sun).value, 0.01) * u.R_sun
    else:
        return r.to(u.R_sun)


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


def bin_edge_summary(r, binfit='left'):
    """

    """
    if binfit == 'center':
        rfit = 0.5*(r[0:-1] + r[1:])
    elif binfit == 'left':
        rfit = r[0:-1]
    elif binfit == 'right':
        rfit = r[1:]
    else:
        raise ValueError('Keyword "binfit" must have value "center", "left" or "right"')
    return rfit


def fit_radial_intensity(smap, r, fit_range=None, degree=1,
                         binfit='center', summary=np.mean, **summary_kwargs):
    """
    Fits a polynomial function to the natural logarithm of an estimate of the
    intensity as a function of radius.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy Map

    r : `~astropy.units.Quantity`
        A one-dimensional array of bin edges.

    fit_range : None, `~astropy.units.Quantity`


    degree :


    binfit :


    summary :


    summary_kwargs :

    Returns
    -------

    """

    # Get the emission as a function of intensity.
    radial_emission = get_intensity_summary(smap, r, summary=summary, **summary_kwargs)

    # The radial emission is found by calculating a summary statistic of the
    # intensity in bins with bin edges defined by the input 'r'. If 'r' has N
    # values, the radial emission returned has N-1 values.  In order to perform
    # the fit a summary of the bins must be calculated.
    rfit = bin_edge_summary(r, binfit=binfit)

    # Calculate which radii are to be fit.
    if fit_range is None:
        lower = np.min(r)
        upper = np.max(r)
    else:
        lower = fit_range[0]
        upper = fit_range[1]
    fit_here = np.logical_and(lower < rfit, rfit < upper)

    # Radial emission must be above zero
    fit_here = np.logical_and(fit_here, radial_emission > 0)

    # Return the values of a polynomial fit to the log of the radial emission
    return np.polyfit(rfit[fit_here], np.log(radial_emission[fit_here]), degree)


def get_intensity_summary(smap, r, scale=None, summary=np.mean, **summary_kwargs):
    """
    Get a summary statistic of the intensity in a map as a function of radius.

    Parameters
    ----------
    smap : sunpy.map.Map
        A sunpy map.

    r : `~astropy.units.Quantity`
        A one-dimensional array of bin edges.

    scale : None, `~astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.

    Returns
    -------
    intensity summary : `~numpy.array`
        A summary statistic of the radial intensity in the bins defined by the
        bin edges.  If "r" has N bins, the returned array has N-1 values.
    """
    if scale is None:
        s = smap.rsun_obs
    else:
        s = scale

    # Get the radial distance of every pixel from the center of the Sun.
    map_r = find_pixel_radii(smap, scale=s).to(u.R_sun)

    # Calculate the summary statistic in the radial bins.
    return np.asarray([summary(smap.data[(map_r > r[i].to(u.R_sun)) * (map_r < r[i+1].to(u.R_sun))], **summary_kwargs) for i in range(0, r.size-1)])


def compensate(radii, p, normalize=True, normalization_radius=1*u.R_sun):
    """
    Calculate the compensation factors at the input radii.

    Parameters
    ----------
    radii :
    p :
    normalize : bool
        The compensation factor is 1 at the normalization radius

    normalization_radius : `~astropy.units.Quantity`
        The radius at which the compensation factor is 1.

    Returns
    -------

    """
    these_radii = radii.to(u.R_sun).value

    if normalize:
        nr = normalization_radius.to(u.R_sun).value
        polynomial = np.poly1d(p)(these_radii) - np.poly1d(p)(nr)
    else:
        polynomial = np.poly1d(p)(these_radii)
    return 1 / np.exp(polynomial)


def offlimb_intensity_enhance(smap, solar_radius=None, r=None, degree=1,
                              normalization_radius=1*u.R_sun,
                              annular_function=np.mean,
                              **annular_function_kwargs):
    """
    A convenience object that calculates

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy map

    solar_radius : None or `~astropy.units.Quantity`
        If None, the map is queried for the radius of the Sun in map units.
        For example, in typical helioprojective Cartesian maps the solar
        radius is expressed in units of arcseconds.

    r : `~astropy.units.Quantity`
        A one-dimensional array of bin edges.

    degree : int


    normalization_radius : `~astropy.units.Quantity`


    annular_function :


    annular_function_kwargs :


    """

    # Define the radius of the Sun
    if solar_radius is None:
        rsun_obs = smap.rsun_obs
    else:
        rsun_obs = solar_radius

    # Get the radii for every pixel
    map_r = find_pixel_radii(smap, scale=rsun_obs).to(u.R_sun)

    # Which radii are we going to fit?
    radial_bins = _radial_bins(r, map_r)

    # Fits a polynomial function to the natural logarithm of an estimate of
    # the intensity as a function of radius.
    fit_polynomial = fit_radial_intensity(smap, radial_bins, degree=degree,
                                          summary=annular_function,
                                          **annular_function_kwargs)

    # Calculate the compensation function
    compensation = compensate(map_r, fit_polynomial, normalize=True,
                              normalization_radius=normalization_radius)
    compensation[map_r < normalization_radius] = 1

    return sunpy.map.Map(smap.data * compensation, smap.meta)


def nrgf(smap, annular_function=np.mean, r=None, width_function=np.std,
         application_radius=1*u.R_sun, intensity_scale=True):
    """
    Implementation of the normalizing radial gradient filter of
    Morgan, Habbal & Woo, 2006, Sol. Phys., 236, 263.
    https://link.springer.com/article/10.1007%2Fs11207-006-0113-6

    Parameters
    ----------
    smap
    annular_function
    r
    width_function
    application_radius
    intensity_scale

    Returns
    -------

    """

    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # Which radii are we going to fit?
    radial_bins = _radial_bins(r, map_r)

    #
    if intensity_scale:
        inside = map_r <= application_radius.to(u.R_sun)
        min_value = np.min(smap.data[inside])
        max_value = np.max(smap.data[inside])


    intensity_summary = get_intensity_summary(smap, radial_bins, summary=annular_function)

    intensity_distribution_summary = get_intensity_summary(smap, radial_bins, scale=None, summary=width_function)

    data = np.zeros_like(smap.data)
    for i in range(0, len(radial_bins)-1):
        here = np.logical_and(map_r > radial_bins[i], map_r < radial_bins[i+1])
        here = np.logical_and(here, map_r > application_radius)
        data[here] = (smap.data[here] - intensity_summary[i]) / intensity_distribution_summary[i]

    return sunpy.map.Map(data, smap.meta)
