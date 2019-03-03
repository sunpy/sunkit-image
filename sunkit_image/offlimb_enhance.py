"""
This module contains functions that can be used to enchance the regions off the solar limb.
"""
import numpy as np
import astropy.units as u

import sunpy.map

from sunkit_image.utils.utils import (
    bin_edge_summary,
    find_pixel_radii,
    get_radial_intensity_summary,
)

__all__ = [
    "fit_polynomial_to_log_radial_intensity",
    "calculate_fit_radial_intensity",
    "normalize_fit_radial_intensity",
    "intensity_enhance",
    "normalizing_radial_gradient_filter",
]


def fit_polynomial_to_log_radial_intensity(radii, intensity, degree):
    """
    Fits a polynomial of degree, "degree" to the log of the radial intensity.

    Parameters
    ----------
    radii : `astropy.units.Quantity`
        The radii at which the fitted intensity is calculated, nominally in
        units of solar radii.
    intensity : `numpy.ndarray`
        The 1D intensity array to fit the polynomial for.
    degree : `int`
        The degree of the polynomial.

    Returns
    -------
    polynomial : `numpy.ndarray`
        A polynomial of degree, "degree" that fits the log of the intensity
        profile as a function of the radius.
    """
    return np.polyfit(radii.to(u.R_sun).value, np.log(intensity), degree)


def calculate_fit_radial_intensity(radii, polynomial):
    """
    Calculates the fit value of the radial intensity at the values "radii". The function assumes
    that the polynomial is the best fit to the observed log of the intensity as a function of
    radius.

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
    Normalizes the fitted radial intensity to the value at the normalization radius. The function
    assumes that the polynomial is the best fit to the observed log of the intensity as a function
    of radius.

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
    return calculate_fit_radial_intensity(
        radii, polynomial
    ) / calculate_fit_radial_intensity(normalization_radius, polynomial)


def intensity_enhance(
    smap,
    radial_bin_edges,
    scale=None,
    summarize_bin_edges="center",
    summary=np.mean,
    degree=1,
    normalization_radius=1 * u.R_sun,
    fit_range=[1, 1.5] * u.R_sun,
    **summary_kwargs
):
    """
    Returns a map with the offlimb emission enhanced.

    The enhancement is calculated as follows:

    A summary statistic of the radial dependence of the offlimb emission is calculated.
    Since the UV and EUV emission intensity drops of quickly off the solar limb, it makes sense
    to fit the log of the intensity statistic using some appropriate function.
    The function we use here is a polynomial.
    To calculate the enhancement, the fitted function is normalized to its value at the
    normalization radius from the center of the Sun (a sensible choice is the solar radius).
    The offlimb emission is then divided by this normalized function.

    .. note::

        After applying the filter, current plot settings such as the image normalization
        may have to be changed in order to obtain a good-looking plot.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        The SunPy map to enchance.
    radial_bin_edges : `astropy.units.Quantity`
        A two-dimensional array of bin edges of size [2, nbins] where nbins is
        the number of bins.
    scale : None or `astropy.units.Quantity`, optional
        The radius of the Sun expressed in map units.
        For example, in typical Helioprojective Cartesian maps the solar radius is expressed in
        units of arcseconds.
        Defaults to None, which means that the map scale is used.
    summarize_bin_edges : `str`, optional
        How to summarize the bin edges.s
        Defaults to "center".
    summary : `function`, optional
        A function that returns a summary statistic of the radial intensity.
        Defaults to `numpy.mean`.
    degree : `int`, optional
        Degree of the polynomial fit to the log of the intensity as a function of radius.
        Defaults to 1.
    normalization_radius : `astropy.units.Quantity`, optional
        The radius at which the enhancement has value 1.
        For most cases the value of the enhancement will increase as a function of radius.
        Defaults to 1 solar radii.
    fit_range : `astropy.units.Quantity`, optional
        Array-like with 2 elements defining the range of radii over which the
        polynomial function is fit.
        The preferred units are solar radii.
        Defaults to [1, 1.5] solar radii.
    summary_kwargs : `dict`, optional
        Keywords applicable to the summary function.

    Returns
    -------
    new_map : `sunpy.map.Map`
        A SunPy map that has the emission above the normalization radius enhanced.
    """
    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # Get the radial intensity distribution
    radial_intensity = get_radial_intensity_summary(
        smap, radial_bin_edges, scale=scale, summary=summary, **summary_kwargs
    )

    # Summarize the radial bins
    radial_bin_summary = bin_edge_summary(radial_bin_edges, summarize_bin_edges).to(
        u.R_sun
    )

    # Fit range
    if fit_range[0] >= fit_range[1]:
        raise ValueError("The fit range must be strictly increasing.")

    fit_here = np.logical_and(
        fit_range[0].to(u.R_sun).value <= radial_bin_summary.to(u.R_sun).value,
        radial_bin_summary.to(u.R_sun).value <= fit_range[1].to(u.R_sun).value,
    )

    # Fits a polynomial function to the natural logarithm of an estimate of
    # the intensity as a function of radius.
    polynomial = fit_polynomial_to_log_radial_intensity(
        radial_bin_summary[fit_here], radial_intensity[fit_here], degree
    )

    # Calculate the enhancement
    enhancement = 1 / normalize_fit_radial_intensity(
        map_r, polynomial, normalization_radius
    )
    enhancement[map_r < normalization_radius] = 1

    # Return a map with the intensity enhanced above the normalization radius
    # and the same meta data as the input map.
    return sunpy.map.Map(smap.data * enhancement, smap.meta)


def normalizing_radial_gradient_filter(
    smap,
    radial_bin_edges,
    scale=None,
    intensity_summary=np.mean,
    intensity_summary_kwargs=None,
    width_function=np.std,
    width_function_kwargs=None,
    application_radius=1 * u.R_sun,
):
    """
    Implementation of the normalizing radial gradient filter (NRGF).

    .. note::

        After applying the filter, current plot settings such as the image normalization
        may have to be changed in order to obtain a good-looking plot.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        The SunPy map to enchance.
    radial_bin_edges : `astropy.units.Quantity`
        A two-dimensional array of bin edges of size [2, nbins] where nbins is
        the number of bins.
    scale : None or `astropy.units.Quantity`, optional
        The radius of the Sun expressed in map units.
        For example, in typical Helioprojective Cartesian maps the solar radius is expressed in
        units of arcseconds.
        Defaults to None, which means that the map scale is used.
    summarize_bin_edges : `str`, optional
        How to summarize the bin edges.s
        Defaults to "center".
    intensity_summary : `function`, optional
        A function that returns a summary statistic of the radial intensity.
        Defaults to `numpy.mean`.
    intensity_summary_kwargs : None, `dict`
        Keywords applicable to the summary function.
    width_function : `function`
        A function that returns a summary statistic of the distribution of intensity,
        at a given radius.
        Defaults to `numpy.std`.
    width_function_kwargs : `function`
        Keywords applicable to the width function.
    application_radius : `astropy.units.Quantity`
        The NRGF is applied to emission at radii above the application_radius.
        Defaults to 1 solar radii.

    Returns
    -------
    new_map : `sunpy.map.Map`
        A SunPy map that has had the NRGF applied to it.

    References
    ----------
    * Morgan, Habbal & Woo, 2006, Sol. Phys., 236, 263. https://link.springer.com/article/10.1007%2Fs11207-006-0113-6
    """

    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # Radial intensity
    radial_intensity = get_radial_intensity_summary(
        smap,
        radial_bin_edges,
        scale=scale,
        summary=intensity_summary,
        **intensity_summary_kwargs
    )

    # An estimate of the width of the intensity distribution in each radial bin.
    radial_intensity_distribution_summary = get_radial_intensity_summary(
        smap,
        radial_bin_edges,
        scale=scale,
        summary=width_function,
        **width_function_kwargs
    )

    # Storage for the filtered data
    data = np.zeros_like(smap.data)

    # Calculate the filter for each radial bin.
    for i in range(0, radial_bin_edges.shape[1]):
        here = np.logical_and(
            map_r > radial_bin_edges[0, i], map_r < radial_bin_edges[1, i]
        )
        here = np.logical_and(here, map_r > application_radius)
        data[here] = (
            smap.data[here] - radial_intensity[i]
        ) / radial_intensity_distribution_summary[i]

    return sunpy.map.Map(data, smap.meta)
