"""
This module contains functions that can be used to enhance the regions above a
radius.
"""
import numpy as np

import astropy.units as u
import sunpy.map
from sunpy.coordinates import frames

from sunkit_image.utils import (
    bin_edge_summary,
    equally_spaced_bins,
    find_pixel_radii,
    get_radial_intensity_summary,
)

__all__ = [
    "fit_polynomial_to_log_radial_intensity",
    "calculate_fit_radial_intensity",
    "normalize_fit_radial_intensity",
    "intensity_enhance",
    "nrgf",
    "set_attenuation_coefficients",
    "fnrgf",
]


def fit_polynomial_to_log_radial_intensity(radii, intensity, degree):
    """
    Fits a polynomial of  a given degree to the log of the radial intensity.

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
    `numpy.ndarray`
        A polynomial of degree, ``degree`` that fits the log of the intensity
        profile as a function of the radius.
    """
    return np.polyfit(radii.to(u.R_sun).value, np.log(intensity), degree)


def calculate_fit_radial_intensity(radii, polynomial):
    """
    Calculates the fit value of the radial intensity at the values ``radii``.

    The function assumes that the polynomial is the best fit to the observed
    log of the intensity as a function of radius.

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
    `numpy.ndarray`
        An array with the same shape as radii which expresses the fitted
        intensity value.
    """
    return np.exp(np.poly1d(polynomial)(radii.to(u.R_sun).value))


def normalize_fit_radial_intensity(radii, polynomial, normalization_radius):
    """
    Normalizes the fitted radial intensity to the value at the normalization
    radius.

    The function assumes that the polynomial is the best fit to the observed
    log of the intensity as a function of radius.

    Parameters
    ----------
    radii : `astropy.units.Quantity`
        The radii at which the fitted intensity is calculated, nominally in
        units of solar radii.
    polynomial : `numpy.ndarray`
        A polynomial of a given degree that fits the log of the intensity
        profile as a function of the radius.
    normalization_radius : `astropy.units.Quantity`
        The radius at which the fitted intensity value is normalized to.

    Returns
    -------
    `numpy.ndarray`
        An array with the same shape as radii which expresses the fitted
        intensity value normalized to its value at the normalization radius.
    """
    return calculate_fit_radial_intensity(radii, polynomial) / calculate_fit_radial_intensity(
        normalization_radius, polynomial
    )


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
    Returns a SunPy Map with the intensity enhanced above a given radius.

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
        A two-dimensional array of bin edges of size ``[2, nbins]`` where ``nbins`` is
        the number of bins.
    scale : `astropy.units.Quantity`, optional
        The radius of the Sun expressed in map units.
        For example, in typical Helioprojective Cartesian maps the solar radius is expressed in
        units of arcseconds.
        Defaults to None, which means that the map scale is used.
    summarize_bin_edges : `str`, optional
        How to summarize the bin edges.
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
        Defaults to ``[1, 1.5]`` solar radii.
    summary_kwargs : `dict`, optional
        Keywords applicable to the summary function.

    Returns
    -------
    `sunpy.map.Map`
        A SunPy map that has the emission above the normalization radius enhanced.
    """
    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # Get the radial intensity distribution
    radial_intensity = get_radial_intensity_summary(
        smap, radial_bin_edges, scale=scale, summary=summary, **summary_kwargs
    )

    # Summarize the radial bins
    radial_bin_summary = bin_edge_summary(radial_bin_edges, summarize_bin_edges).to(u.R_sun)

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
    enhancement = 1 / normalize_fit_radial_intensity(map_r, polynomial, normalization_radius)
    enhancement[map_r < normalization_radius] = 1

    # Return a map with the intensity enhanced above the normalization radius
    # and the same meta data as the input map.
    return sunpy.map.Map(smap.data * enhancement, smap.meta)


def nrgf(
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
    Implementation of the normalizing radial gradient filter (NRGF).

    The filter works as follows:

    Normalizing Radial Gradient Filter a simple filter for removing the radial gradient to reveal
    coronal structure. Applied to polarized brightness observations of the corona, the NRGF produces
    images which are striking in their detail. It takes the input map and find the intensity summary
    and width of intenstiy values in radial bins above the application radius. The intensity summary
    and the width is then used to normalize the intensity values in a particular radial bin.

    .. note::

        After applying the filter, current plot settings such as the image normalization
        may have to be changed in order to obtain a good-looking plot.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        The SunPy map to enchance.
    radial_bin_edges : `astropy.units.Quantity`
        A two-dimensional array of bin edges of size ``[2, nbins]`` where ``nbins`` is
        the number of bins.
    scale : None or `astropy.units.Quantity`, optional
        The radius of the Sun expressed in map units.
        For example, in typical Helioprojective Cartesian maps the solar radius is expressed in
        units of arcseconds.
        Defaults to None, which means that the map scale is used.
    intensity_summary : `function`, optional
        A function that returns a summary statistic of the radial intensity.
        Defaults to `numpy.nanmean`.
    intensity_summary_kwargs : `dict`, optional
        Keywords applicable to the summary function.
    width_function : `function`, optional
        A function that returns a summary statistic of the distribution of intensity,
        at a given radius.
        Defaults to `numpy.std`.
    width_function_kwargs : `function`, optional
        Keywords applicable to the width function.
    application_radius : `astropy.units.Quantity`, optional
        The NRGF is applied to emission at radii above the application_radius.
        Defaults to 1 solar radii.

    Returns
    -------
    `sunpy.map.Map`
        A SunPy map that has had the NRGF applied to it.

    References
    ----------
    * Morgan, Habbal & Woo, 2006, Sol. Phys., 236, 263.
      https://link.springer.com/article/10.1007%2Fs11207-006-0113-6
    """

    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # To make sure bins are in the map.
    if radial_bin_edges[1, -1] > np.max(map_r):
        radial_bin_edges = equally_spaced_bins(
            inner_value=radial_bin_edges[0, 0], outer_value=np.max(map_r), nbins=radial_bin_edges.shape[1]
        )

    # Radial intensity
    radial_intensity = get_radial_intensity_summary(
        smap, radial_bin_edges, scale=scale, summary=intensity_summary, **intensity_summary_kwargs
    )

    # An estimate of the width of the intensity distribution in each radial bin.
    radial_intensity_distribution_summary = get_radial_intensity_summary(
        smap, radial_bin_edges, scale=scale, summary=width_function, **width_function_kwargs
    )

    # Storage for the filtered data
    data = np.zeros_like(smap.data)

    # Calculate the filter value for each radial bin.
    for i in range(0, radial_bin_edges.shape[1]):
        here = np.logical_and(map_r >= radial_bin_edges[0, i], map_r < radial_bin_edges[1, i])
        here = np.logical_and(here, map_r > application_radius)
        data[here] = smap.data[here] - radial_intensity[i]
        if radial_intensity_distribution_summary[i] != 0.0:
            data[here] = data[here] / radial_intensity_distribution_summary[i]

    return sunpy.map.Map(data, smap.meta)


def set_attenuation_coefficients(order, range_mean=[1.0, 0.0], range_std=[1.0, 0.0], cutoff=0):
    """
    This is a helper function to Fourier Normalizing Radial Gradient Filter
    (`sunkit_image.radial.fnrgf`).

    This function sets the attenuation coefficients in the one of the following two manners:

    If ``cutoff`` is ``0``, then it will set the attenuation coefficients as linearly decreasing between
    the range ``range_mean`` for the attenuation coefficents for mean approximation and ``range_std`` for
    the attenuation coefficients for standard deviation approximation.

    If ``cutoff`` is not ``0``, then it will set the last ``cutoff`` number of coefficients equal to zero
    while all the others the will be set as linearly decreasing as described above.

    .. note::

        This function only describes some of the ways in which attenuation coefficients can be calculated.
        The optimal coefficients depends on the size and quality of image. There is no generalized formula
        for choosing them and its upto the user to choose a optimum value.

    Parameters
    ----------
    order : `int`
        The order of the Fourier approximation.
    range_mean : `list`, optional
        A list of length of ``2`` which contains the highest and lowest values between which the coefficients for
        mean approximation be calculated in a linearly decreasing manner.
    range_std : `list`, optional
        A list of length of ``2`` which contains the highest and lowest values between which the coefficients for
        standard deviation approximation be calculated in a linearly decreasing manner.
    cutoff : `int`, optional
        The numbers of coefficients from the last that should be set to ``zero``.

    Returns
    -------
    `numpy.ndarray`
        A numpy array of shape ``[2, order + 1]`` containing the attenuation coefficients for the Fourier
        coffiecients. The first row describes the attenustion coefficients for the Fourier coefficients of
        the mean approximation. The second row contains the attenuation coefficients for the Fourier coefficients
        of the standard deviation approximation.
    """

    attenuation_coefficients = np.zeros((2, order + 1))
    attenuation_coefficients[0, :] = np.linspace(range_mean[0], range_mean[1], order + 1)
    attenuation_coefficients[1, :] = np.linspace(range_std[0], range_std[1], order + 1)

    if cutoff > (order + 1):
        raise ValueError("Cutoff cannot be greater than order + 1.")

    if cutoff != 0:
        attenuation_coefficients[:, (-1 * cutoff) :] = 0

    return attenuation_coefficients


def fnrgf(
    smap,
    radial_bin_edges,
    order,
    attenuation_coefficients,
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
    Implementation of the fourier normalizing radial gradient filter (FNRGF).

    The filter works as follows:

    Fourier Normalizing Radial Gradient Filter approximates the local average and the local standard
    deviation by a finite Fourier series. This method enables the enhancement of finer details, especially
    in regions of lower contrast. It takes the input map and divides the region above the application
    radius and in the radial bins into various small angular segments. Then for each of these angular
    segments, the intensity summary and width is calculated. The intensity summary and the width of each
    angular segments are then used to find a Fourier approximation of the intensity summary and width for
    the entire radial bin, this Fourier approximated value is then used to noramlize the intensity in the
    radial bin.

    .. note::

        After applying the filter, current plot settings such as the image normalization
        may have to be changed in order to obtain a good-looking plot.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy map.
    radial_bin_edges : `astropy.units.Quantity`
        A two-dimensional array of bin edges of size ``[2, nbins]`` where ``nbins`` is the number of bins.
    order : `int`
        Order (number) of fourier coefficients and it can not be lower than 1.
    attenuation_coefficients : `float`
        A two dimensional array of shape ``[2, order + 1]``. The first row contain attenuation
        coefficients for mean calculations. The second row contains attenuation coefficients
        for standard deviation calculation.
    ratio_mix : `float`, optional
        A one dimensional array of shape ``[2, 1]`` with values equal to ``[K1, K2]``.
        The ratio in which the original image and filtered image are mixed.
        Defaults to ``[15, 1]``.
    scale : `None` or `astropy.units.Quantity`, optional
        The radius of the Sun expressed in map units. For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds. If `None` (the default), then the map scale is used.
    intensity_summary :`function`, optional
        A function that returns a summary statistic of the radial intensity.
        Default is `numpy.nanmean`.
    intensity_summary_kwargs : `None`, `~dict`
        Keywords applicable to the summary function.
    width_function : `function`
        A function that returns a summary statistic of the distribution of intensity, at a given radius.
        Defaults to `numpy.std`.
    width_function_kwargs : `function`
        Keywords applicable to the width function.
    application_radius : `astropy.units.Quantity`
        The FNRGF is applied to emission at radii above the application_radius.
        Defaults to 1 solar radii.
    number_angular_segments : `int`
        Number of angular segments in a circular annulus.
        Defaults to 130.

    Returns
    -------
    `sunpy.map.Map`
        A SunPy map that has had the FNRGF applied to it.

    References
    ----------
    * Morgan, Habbal & Druckmüllerová, 2011, Astrophysical Journal 737, 88.
      https://iopscience.iop.org/article/10.1088/0004-637X/737/2/88/pdf.
    * The implementation is highly inspired by this doctoral thesis.
      DRUCKMÜLLEROVÁ, H. Application of adaptive filters in processing of solar corona images.
      https://dspace.vutbr.cz/bitstream/handle/11012/34520/DoctoralThesis.pdf.
    """

    if order < 1:
        raise ValueError("Minimum value of order is 1")

    # Get the radii for every pixel
    map_r = find_pixel_radii(smap).to(u.R_sun)

    # To make sure bins are in the map.
    if radial_bin_edges[1, -1] > np.max(map_r):
        radial_bin_edges = equally_spaced_bins(
            inner_value=radial_bin_edges[0, 0], outer_value=np.max(map_r), nbins=radial_bin_edges.shape[1]
        )

    # Get the Helioprojective coordinates of each pixel
    x, y = np.meshgrid(*[np.arange(v.value) for v in smap.dimensions]) * u.pix
    coords = smap.pixel_to_world(x, y).transform_to(frames.Helioprojective)

    # Get angles associated with every pixel
    angles = np.arctan2(coords.Ty.value, coords.Tx.value)

    # Making sure all angles are between (0, 2 * pi)
    angles = np.where(angles < 0, angles + (2 * np.pi), angles)

    # Number of radial bins
    nbins = radial_bin_edges.shape[1]

    # Storage for the filtered data
    data = np.zeros_like(smap.data)

    # Iterate over each circular ring
    for i in range(0, nbins):

        # Finding the pixels which belong to a certain circular ring
        annulus = np.logical_and(map_r >= radial_bin_edges[0, i], map_r < radial_bin_edges[1, i])
        annulus = np.logical_and(annulus, map_r > application_radius)

        # The angle subtended by each segment
        segment_angle = 2 * np.pi / number_angular_segments

        # Storage of mean and standard deviation of each segment in a circular ring
        average_segments = np.zeros((1, number_angular_segments))
        std_dev = np.zeros((1, number_angular_segments))

        # Calculating sin and cos of the angles to be multiplied with the means and standard
        # deviations to give the fourier approximation
        cos_matrix = np.cos(
            np.array(
                [
                    [(2 * np.pi * (j + 1) * (i + 0.5)) / number_angular_segments for j in range(order)]
                    for i in range(number_angular_segments)
                ]
            )
        )
        sin_matrix = np.sin(
            np.array(
                [
                    [(2 * np.pi * (j + 1) * (i + 0.5)) / number_angular_segments for j in range(order)]
                    for i in range(number_angular_segments)
                ]
            )
        )

        # Iterate over each segment in a circular ring
        for j in range(0, number_angular_segments, 1):

            # Finding all the pixels whose angle values lie in the segment
            angular_segment = np.logical_and(angles >= segment_angle * j, angles < segment_angle * (j + 1))

            # Finding the particular segment in the circular ring
            annulus_segment = np.logical_and(annulus, angular_segment)

            # Finding mean and standard deviation in each segnment. If takes care of the empty
            # slices.
            if np.sum([annulus_segment > 0]) == 0:
                average_segments[0, j] = 0
                std_dev[0, j] = 0
            else:
                average_segments[0, j] = intensity_summary(smap.data[annulus_segment])
                std_dev[0, j] = width_function(smap.data[annulus_segment])

        # Calculating the fourier coefficients multiplied with the attenuation coefficients
        # Refer to equation (2), (3), (4), (5) in the paper
        fourier_coefficient_a_0 = np.sum(average_segments) * (2 / number_angular_segments)
        fourier_coefficient_a_0 *= attenuation_coefficients[0, 1]

        fourier_coefficients_a_k = np.matmul(average_segments, cos_matrix) * (2 / number_angular_segments)
        fourier_coefficients_a_k *= attenuation_coefficients[0][1:]

        fourier_coefficients_b_k = np.matmul(average_segments, sin_matrix) * (2 / number_angular_segments)
        fourier_coefficients_b_k *= attenuation_coefficients[0][1:]

        # Refer to equation (6) in the paper
        fourier_coefficient_c_0 = np.sum(std_dev) * (2 / number_angular_segments)
        fourier_coefficient_c_0 *= attenuation_coefficients[1, 1]

        fourier_coefficients_c_k = np.matmul(std_dev, cos_matrix) * (2 / number_angular_segments)
        fourier_coefficients_c_k *= attenuation_coefficients[1][1:]

        fourier_coefficients_d_k = np.matmul(std_dev, sin_matrix) * (2 / number_angular_segments)
        fourier_coefficients_d_k *= attenuation_coefficients[1][1:]

        # To calculate the multiples of angles of each pixel for finding the fourier approximation
        # at that point. See equations 6.8 and 6.9 of the doctoral thesis.
        K_matrix = np.ones((order, np.sum(annulus > 0))) * np.array(range(1, order + 1)).T.reshape(order, 1)
        phi_matrix = angles[annulus].reshape((1, angles[annulus].shape[0]))
        angles_of_pixel = K_matrix * phi_matrix

        # Get the approxiamted value of mean
        mean_approximated = np.matmul(fourier_coefficients_a_k, np.cos(angles_of_pixel))
        mean_approximated += np.matmul(fourier_coefficients_b_k, np.sin(angles_of_pixel))
        mean_approximated += fourier_coefficient_a_0 / 2

        # Get the approxiamted value of standard deviation
        std_approximated = np.matmul(fourier_coefficients_c_k, np.cos(angles_of_pixel))
        std_approximated += np.matmul(fourier_coefficients_d_k, np.sin(angles_of_pixel))
        std_approximated += fourier_coefficient_c_0 / 2

        # Normailize the data
        # Refer equation (7) in the paper
        std_approximated = np.where(std_approximated == 0.00, 1, std_approximated)
        data[annulus] = np.ravel((smap.data[annulus] - mean_approximated) / std_approximated)

        # Linear combination of original image and the filtered data.
        data[annulus] = ratio_mix[0] * smap.data[annulus] + ratio_mix[1] * data[annulus]

    return sunpy.map.Map(data, smap.meta)
