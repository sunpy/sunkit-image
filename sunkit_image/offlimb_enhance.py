"""
This package contains functions that can be used to enchance the regions off the solar limb.
"""
import numpy as np

import astropy.units as u

import sunpy.map

from sunpy.coordinates import frames

from sunkit_image.utils.utils import (
    find_pixel_radii,
    bin_edge_summary,
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
    Returns a map with the off-limb emission enhanced.  The enhancement is calculated as follows.  A
    summary statistic of the radial dependence of the off-limb emission is calculated.  Since the UV
    and EUV emission intensity drops of quickly off the solar limb, it makes sense to fit the log of
    the intensity statistic using some appropriate function.  The function we use here is a
    polynomial.  To calculate the enhancement, the fitted function is normalized to its value at the
    normalization radius from the center of the Sun (a sensible choice is the solar radius).  The
    offlimb emission is then divided by this normalized function.

    Note that after enhancement plot settings such as the image normalization
    may have to be changed in order to obtain a good-looking plot.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy map

    radial_bin_edges : `astropy.units.Quantity`
        A two-dimensional array of bin edges of size [2, nbins] where nbins is
        the number of bins.

    scale : None | `astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None, then the map scale is used.

    summarize_bin_edges : `str`
        How to summarize the bin edges.

    summary : `function`
        A function that returns a summary statistic of the radial intensity,
        for example `~numpy.mean` and `~numpy.median`.

    degree : `int`
        Degree of the polynomial fit to the log of the intensity as a function of
        radius.

    normalization_radius : `astropy.units.Quantity`
        The radius at which the enhancement has value 1.  For most cases
        the value of the enhancement will increase as a function of
        radius.

    fit_range : `astropy.units.Quantity`
        Array like with 2 elements defining the range of radii over which the
        polynomial function is fit.  The preferred units are solar radii.

    summary_kwargs : `dict`
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
    intensity_summary=np.nanmean,
    intensity_summary_kwargs={},
    width_function=np.std,
    width_function_kwargs={},
    application_radius=1 * u.R_sun,
):
    """
    Implementation of the normalizing radial gradient filter (NRGF) of Morgan, Habbal & Woo, 2006,
    Sol. Phys., 236, 263. https://link.springer.com/article/10.1007%2Fs11207-006-0113-6.

    Note that after applying the NRGF plot settings such as the image normalization
    may have to be changed in order to obtain a good-looking plot.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy map

    radial_bin_edges : `astropy.units.Quantity`
        A two-dimensional array of bin edges of size [2, nbins] where nbins is
        the number of bins.

    scale : None | `astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None, then the map scale is used.

    intensity_summary :`function`
        A function that returns a summary statistic of the radial intensity,
        for example `~numpy.mean` and `~numpy.median`.

    intensity_summary_kwargs : None | `dict`
        Keywords applicable to the summary function.

    width_function : `function`
        A function that returns a summary statistic of the distribution of intensity,
        at a given radius, for example `~numpy.std`.

    width_function_kwargs : `function`
        Keywords applicable to the width function.

    application_radius : `astropy.units.Quantity`
        The NRGF is applied to emission at radii above the application_radius.

    Returns
    -------
    new_map : `sunpy.map.Map`
        A SunPy map that has had the NRGF applied to it.
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


def fourier_normalizing_radial_gradient_filter(
    smap,
    radial_bin_edges,
    order=30,
    ratio_mix=[15, 1],
    scale=None,
    intensity_summary=np.nanmean,
    intensity_summary_kwargs={},
    width_function=np.std,
    width_function_kwargs={},
    application_radius=1 * u.R_sun,
    number_angular_segments=130,
):
    """
    Implementation of the fourier normalizing radial gradient filter (FNRGF) of Morgan, Habbal & Druckmüllerová, 2011,
    Astrophysical Journal 737, 88. https://iopscience.iop.org/article/10.1088/0004-637X/737/2/88/pdf.

    The implementation is highly inspired by https://dspace.vutbr.cz/bitstream/handle/11012/34520/DoctoralThesis.pdf.

    Note that after applying the FNRGF plot settings such as the image normalization
    may have to be changed in order to obtain a good-looking plot.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy map

    radial_bin_edges : `~astropy.units.Quantity`
        A two-dimensional array of bin edges of size [2, nbins] where nbins is
        the number of bins.

    order : `~int`
        Order (Number) of fourier coefficients.

    ratio_mix : `~float`
        The ratio in which the original image and filtered image are mixed.

    scale : None | `astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.  If None, then the map scale is used.

    intensity_summary :`~function`
        A function that returns a summary statistic of the radial intensity,
        for example `~numpy.mean` and `~numpy.median`.

    intensity_summary_kwargs : None | `~dict`
        Keywords applicable to the summary function.

    width_function : `~function`
        A function that returns a summary statistic of the distribution of intensity,
        at a given radius, for example `~numpy.std`.

    width_function_kwargs : `~function`
        Keywords applicable to the width function.

    application_radius : `~astropy.units.Quantity`
        The FNRGF is applied to emission at radii above the application_radius.

    number_angular_segments : `~int`
        Number of angular segments in a circular annulus. Default value is 130. It should always be
        even.

    Returns
    -------
    new_map : `sunpy.map.Map`
        A SunPy map that has had the FNRGF applied to it.
    """

    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # Get the Helioprojective coordinates of each pixel
    x, y = np.meshgrid(*[np.arange(v.value) for v in smap.dimensions]) * u.pix
    coords = smap.pixel_to_world(x, y).transform_to(frames.Helioprojective)

    # Get angles associated with every pixel
    angles = np.arctan2(coords.Ty.value, coords.Tx.value)

    # Making sure all angles are between (0, 2 * pi)
    angles = np.where(angles < 0, angles + (2*np.pi), angles)

    # Number of radial bins
    nbins = radial_bin_edges.shape[1]

    # Storage for the filtered data
    data = np.zeros_like(smap.data)

    # Iterate over each circular ring
    for i in range(0, nbins):

        # Finding the pixels which belong to a certain circular ring
        annulus = np.logical_and(map_r > radial_bin_edges[0, i], map_r < radial_bin_edges[1, i])
        annulus = np.logical_and(annulus, map_r > application_radius)

        # The angle subtended by each segment
        segment_angle = 2*np.pi / number_angular_segments

        # Storage of mean and standard deviation of each segment in a circular ring
        average_segments = np.zeros((1, number_angular_segments))
        std_dev = np.zeros((1, number_angular_segments))

        # Calculating sin and cos of the angles to be multiplied with the means and standard
        # deviations to give the fourier approximation
        cos_matrix = np.cos(np.array([[(2 * np.pi * (j + 1) * (i + 0.5)) / number_angular_segments for j in range(order)] for i in range(number_angular_segments)]))
        sin_matrix = np.sin(np.array([[(2 * np.pi * (j + 1) * (i + 0.5)) / number_angular_segments for j in range(order)] for i in range(number_angular_segments)]))

        # Iterate over each segment in a circular ring
        for j in range(0, number_angular_segments, 1):
       
            # Finding all the pixels whose angle values lie in the segment
            angular_segment = np.logical_and(angles > segment_angle * j, angles < segment_angle * (j + 1))

            # Finding the particular segment in the circular ring
            annulus_segment = np.logical_and(annulus, angular_segment)

            # Finding mean and standard deviation in each segnment. If takes care of the empty slices.
            if np.sum([annulus_segment > 0]) == 0:
                average_segments[0, j] = 0
                std_dev[0, j] = 0
            else:
                average_segments[0, j] = intensity_summary(smap.data[annulus_segment])
                std_dev[0, j] = width_function(smap.data[annulus_segment])

        # Calculating the fourier coefficients
        fourier_coefficient_a_0 = np.sum(average_segments) * (2 / number_angular_segments)
        fourier_coefficients_a_k = np.matmul(average_segments, cos_matrix) * (2 / number_angular_segments)
        fourier_coefficients_b_k = np.matmul(average_segments, sin_matrix) * (2 / number_angular_segments)
        fourier_coefficient_c_0 = np.sum(std_dev) * (2 / number_angular_segments)
        fourier_coefficients_c_k = np.matmul(std_dev, cos_matrix) * (2 / number_angular_segments)
        fourier_coefficients_d_k = np.matmul(std_dev, sin_matrix) * (2 / number_angular_segments)

        # To calculate the multiples of angles of each pixel for finding the fourier approximation
        # at that point. See equations 6.8 and 6.9 of the doctoral thesis.
        K_matrix = np.ones((order, np.sum(annulus > 0))) * np.array(range(1, order+1)).T.reshape(order, 1)
        phi_matrix = angles[annulus].reshape((1, angles[annulus].shape[0]))
        angles_of_pixel = K_matrix * phi_matrix

        # Get the approxiamted value of mean
        mean_approximated = np.matmul(fourier_coefficients_a_k, np.cos(angles_of_pixel)) + np.matmul(fourier_coefficients_b_k, np.sin(angles_of_pixel))
        mean_approximated = mean_approximated + fourier_coefficient_a_0 / 2

        # Get the approxiamted value of standard deviation
        std_approximated = np.matmul(fourier_coefficients_c_k, np.cos(angles_of_pixel)) + np.matmul(fourier_coefficients_d_k, np.sin(angles_of_pixel))
        std_approximated = std_approximated + fourier_coefficient_c_0 / 2

        # Normailize the data
        data[annulus] = np.ravel((smap.data[annulus] - mean_approximated) / std_approximated)

        # Linear combination of original image and the filtered data.
        data[annulus] = ratio_mix[0] * smap.data[annulus] + ratio_mix[1] * data[annulus]

    return sunpy.map.Map(data, smap.meta)
