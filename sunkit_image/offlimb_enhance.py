from __future__ import print_function, division

import numpy as np

import astropy.units as u

import sunpy.map

from sunkit_image.utils.utils import find_pixel_radii, bin_edge_summary, get_radial_intensity_summary


def fit_polynomial_to_log_radial_intensity(radii, intensity, degree):
    """
    Fits a polynomial of degree "degree" to the log of the radial intensity.

    Parameters
    ----------
    radii : `astropy.units.Quantity`


    intensity : `numpy.ndarray`


    degree : `int`


    Returns
    -------
    polynomial : `numpy.ndarray`
        A polynomial of degree "degree" that fits the log of the intensity
        profile as a function of the radius"
    """
    return np.polyfit(radii.to(u.R_sun).value, np.log(intensity), degree)


def calculate_fit_radial_intensity(radii, polynomial):
    """
    Calculates the fit value of the radial intensity at the values "radii". The
    function assumes that the polynomial is the best fit to the observed log of
    the intensity as a function of radius.

    Parameters
    ----------
    radii : `astropy.units.Quantity`
        The radii at which the fitted intensity is calculated, nominally in
        units of solar radii.

    polynomial : `numpy.ndarray`
        A polynomial of degree "degree" that fits the log of the intensity
        profile as a function of the radius.

    Returns
    -------
    fitted intensity : `numpy.ndarray`
        An array with the same shape as radii which expresses the fitted
        intensity value.
    """
    return np.exp(np.poly1d(polynomial)(radii.to(u.R_sun).value))


def normalize_fit_radial_intensity(radii, polynomial, normalization_radius):
    """
    Normalizes the fitted radial intensity to the value at the normalization
    radius. The function assumes that the polynomial is the best fit to the
    observed log of the intensity as a function of radius.

    Parameters
    ----------
    radii : `astropy.units.Quantity`
        The radii at which the fitted intensity is calculated, nominally in
        units of solar radii.

    polynomial : `numpy.ndarray`
        A polynomial of degree "degree" that fits the log of the intensity
        profile as a function of the radius.

    normalization_radius : `astropy.units.Quantity`
        The radius at which the fitted intensity value is normalized to.

    Returns
    -------
    normalized intensity : `numpy.ndarray`
        An array with the same shape as radii which expresses the fitted
        intensity value normalized to its value at the normalization radius.

    """
    return calculate_fit_radial_intensity(radii, polynomial) / calculate_fit_radial_intensity(normalization_radius, polynomial)


def intensity_enhance(smap, radial_bin_edges,
                      scale=None,
                      summarize_bin_edges='center',
                      summary=np.mean,
                      degree=1,
                      normalization_radius=1*u.R_sun,
                      **summary_kwargs):
    """
    Returns a map with the off-limb emission enhanced.  The enhancement
    is calculated as follows.  A summary statistic of the radial dependence
    of the off-limb emission is calculated.  Since the UV and EUV emission
    intensity drops of quickly off the solar limb, it makes sense to fit the
    log of the intensity statistic using some appropriate function.  The
    function we use here is a polynomial.  To calculate the enhancement,
    the fitted function is normalized to its value at the normalization
    radius from the center of the Sun (a sensible choice is the solar
    radius).  The offlimb emission is then divided by this normalized
    function.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy map

    radial_bin_edges : `~astropy.units.Quantity`
        A two-dimensional array of bin edges of size [2, nbins] where nbins is
        the number of bins.

    Keywords
    --------
    scale : None | `astropy.units.Quantity`


    degree : `int`


    summarize_bin_edges :


    summary :


    degree : `int`


    normalization_radius : `~astropy.units.Quantity`


    summary_kwargs : `dict`


    """

    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # Get the radial intensity distribution
    radial_intensity = get_radial_intensity_summary(smap, radial_bin_edges, scale=scale, summary=summary, **summary_kwargs)

    # Summarize the radial bins
    radial_bin_summary = bin_edge_summary(radial_bin_edges, summarize_bin_edges)

    # Fits a polynomial function to the natural logarithm of an estimate of
    # the intensity as a function of radius.
    polynomial = fit_polynomial_to_log_radial_intensity(radial_bin_summary, radial_intensity, degree)

    # Calculate the compensation function
    compensation = 1 / normalize_fit_radial_intensity(map_r, polynomial, normalization_radius)
    compensation[map_r < normalization_radius] = 1

    # Return a map with the intensity enhanced above the normalization radius
    # and the same meta data as the input map.
    return sunpy.map.Map(smap.data * compensation, smap.meta)


def normalizing_radial_gradient_filter(smap, radial_bin_edges,
                                       scale=None,
                                       intensity_summary=np.mean,
                                       width_function=np.std,
                                       application_radius=1*u.R_sun,
                                       **intensity_summary_kwargs):
    """
    Implementation of the normalizing radial gradient filter of
    Morgan, Habbal & Woo, 2006, Sol. Phys., 236, 263.
    https://link.springer.com/article/10.1007%2Fs11207-006-0113-6

    Parameters
    ----------
    smap


    bin_edges


    Keywords
    --------

    :param smap:
    :param radial_bin_edges:
    :param scale:
    :param intensity_summary:
    :param width_function:
    :param application_radius:
    :param intensity_summary_kwargs:
    :return:
    """

    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # Radial intensity
    radial_intensity = get_radial_intensity_summary(smap, radial_bin_edges,
                                                    scale=scale,
                                                    summary=intensity_summary,
                                                    **intensity_summary_kwargs)

    # An estimate of the width of the intensity distribution in each radial bin.
    radial_intensity_distribution_summary = get_radial_intensity_summary(smap, radial_bin_edges,
                                                                         scale=scale,
                                                                         summary=width_function)

    # Storage for the filtered data
    data = np.zeros_like(smap.data)

    # Calculate the filter for each radial bin.
    for i in range(0, radial_bin_edges.shape[1]):
        here = np.logical_and(map_r > radial_bin_edges[0, i], map_r < radial_bin_edges[1, i])
        here = np.logical_and(here, map_r > application_radius)
        data[here] = (smap.data[here] - radial_intensity[i]) / radial_intensity_distribution_summary[i]

    return sunpy.map.Map(data, smap.meta)