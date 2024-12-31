"""
This module contains functions that can be used to enhance the regions above a
radius.
"""

import numpy as np
from tqdm import tqdm

import astropy.units as u

import sunpy.map
from sunpy.coordinates import frames

from sunkit_image.utils import (
    apply_upsilon,
    bin_edge_summary,
    blackout_pixels_above_radius,
    find_radial_bin_edges,
    get_radial_intensity_summary,
)

__all__ = ["fnrgf", "intensity_enhance", "nrgf", "rhef"]


def _fit_polynomial_to_log_radial_intensity(radii, intensity, degree):
    """
    Fits a polynomial of a given degree to the log of the radial intensity.

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


def _calculate_fit_radial_intensity(radii, polynomial):
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


def _normalize_fit_radial_intensity(radii, polynomial, normalization_radius):
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
    return _calculate_fit_radial_intensity(radii, polynomial) / _calculate_fit_radial_intensity(
        normalization_radius,
        polynomial,
    )

def _select_rank_method(method):
    # For now, we have more than one option for ranking the values
    def _percentile_ranks_scipy(arr):   
        from scipy import stats

        return stats.rankdata(arr, method="average") / len(arr)

    def _percentile_ranks_numpy(arr):
        ranks = arr.copy()
        sorted_indices = np.argsort(arr)
        sorted_indices =sorted_indices[~np.isnan(arr[sorted_indices])]
        ranks[sorted_indices] = np.arange(1, len(sorted_indices) + 1)
        return ranks / float(len(sorted_indices))

    def _percentile_ranks_numpy_inplace(arr):
        sorted_indices = np.argsort(arr)
        arr[sorted_indices] = np.arange(1, len(arr) + 1)
        return arr / float(len(arr))

    # Select the sort method
    if method == "inplace":
        ranking_func = _percentile_ranks_numpy_inplace
    elif method == "numpy":
        ranking_func = _percentile_ranks_numpy
    elif method == "scipy":
        ranking_func = _percentile_ranks_scipy
    else:
        msg = f"{method} is invalid. Allowed values are 'inplace', 'numpy', 'scipy'"
        raise NotImplementedError(msg)
    return ranking_func


def intensity_enhance(
    smap,
    *,
    radial_bin_edges=None,
    scale=None,
    summarize_bin_edges="center",
    summary=np.mean,
    degree=1,
    normalization_radius=1 * u.R_sun,
    fit_range=[1, 1.5] * u.R_sun,
    **summary_kwargs,
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

        The returned maps have their ``plot_settings`` changed to remove the extra normalization step.


    Parameters
    ----------
    smap : `sunpy.map.Map`
        The sunpy map to enhance.
    radial_bin_edges : `astropy.units.Quantity`, optional
        A two-dimensional array of bin edges of size ``[2, nbins]`` where ``nbins`` is
        the number of bins.
        Defaults to `None` which will use equally spaced bins.
    scale : `astropy.units.Quantity`, optional
        The radius of the Sun expressed in map units.
        For example, in typical Helioprojective Cartesian maps the solar radius is expressed in
        units of arcseconds.
        Defaults to None, which means that the map scale is used.
    summarize_bin_edges : `str`, optional
        How to summarize the bin edges.
        Defaults to "center".
    summary : ``function``, optional
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
    # Handle the bin edges and radius array
    radial_bin_edges, map_r = find_radial_bin_edges(smap, radial_bin_edges)

    # Get the radial intensity distribution
    radial_intensity = get_radial_intensity_summary(
        smap,
        radial_bin_edges,
        scale=scale,
        summary=summary,
        **summary_kwargs,
    )

    # Summarize the radial bins
    radial_bin_summary = bin_edge_summary(radial_bin_edges, summarize_bin_edges).to(u.R_sun)

    # Fit range
    if fit_range[0] >= fit_range[1]:
        msg = "The fit range must be strictly increasing."
        raise ValueError(msg)

    fit_here = np.logical_and(
        fit_range[0].to(u.R_sun).value <= radial_bin_summary.to(u.R_sun).value,
        radial_bin_summary.to(u.R_sun).value <= fit_range[1].to(u.R_sun).value,
    )

    # Fits a polynomial function to the natural logarithm of an estimate of
    # the intensity as a function of radius.
    polynomial = _fit_polynomial_to_log_radial_intensity(
        radial_bin_summary[fit_here],
        radial_intensity[fit_here],
        degree,
    )

    # Calculate the enhancement
    enhancement = 1 / _normalize_fit_radial_intensity(map_r, polynomial, normalization_radius)
    enhancement[map_r < normalization_radius] = 1

    # Return a map with the intensity enhanced above the normalization radius
    # and the same meta data as the input map.
    new_map = sunpy.map.Map(smap.data * enhancement, smap.meta)
    new_map.plot_settings["norm"] = None
    return new_map


def nrgf(
    smap,
    *,
    radial_bin_edges=None,
    scale=None,
    intensity_summary=np.nanmean,
    intensity_summary_kwargs=None,
    width_function=np.std,
    width_function_kwargs=None,
    application_radius=1 * u.R_sun,
    progress=False,
    fill=np.nan,
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

        The returned maps have their ``plot_settings`` changed to remove the extra normalization step.


    Parameters
    ----------
    smap : `sunpy.map.Map`
        The sunpy map to enhance.
    radial_bin_edges : `astropy.units.Quantity`, optional
        A two-dimensional array of bin edges of size ``[2, nbins]`` where ``nbins`` is
        the number of bins.
        Defaults to `None` which will use equally spaced bins.
    scale : None or `astropy.units.Quantity`, optional
        The radius of the Sun expressed in map units.
        For example, in typical Helioprojective Cartesian maps the solar radius is expressed in
        units of arcseconds.
        Defaults to None, which means that the map scale is used.
    intensity_summary : ``function``, optional
        A function that returns a summary statistic of the radial intensity.
        Defaults to `numpy.nanmean`.
    intensity_summary_kwargs : `dict`, optional
        Keywords applicable to the summary function.
    width_function : ``function``, optional
        A function that returns a summary statistic of the distribution of intensity,
        at a given radius.
        Defaults to `numpy.std`.
    width_function_kwargs : ``function``, optional
        Keywords applicable to the width function.
    application_radius : `astropy.units.Quantity`, optional
        The NRGF is applied to emission at radii above the application_radius.
        Defaults to 1 solar radii.
    progress : `bool`, optional
        Show a progressbar while computing.
        Defaults to `False`.
    fill : Any, optional
        The value to be placed outside of the bounds of the algorithm.
        Defaults to NaN.

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
    if width_function_kwargs is None:
        width_function_kwargs = {}
    if intensity_summary_kwargs is None:
        intensity_summary_kwargs = {}

    # Handle the bin edges and radius array
    radial_bin_edges, map_r = find_radial_bin_edges(smap, radial_bin_edges)

    # Radial intensity
    radial_intensity = get_radial_intensity_summary(
        smap,
        radial_bin_edges,
        scale=scale,
        summary=intensity_summary,
        **intensity_summary_kwargs,
    )

    # An estimate of the width of the intensity distribution in each radial bin.
    radial_intensity_distribution_summary = get_radial_intensity_summary(
        smap,
        radial_bin_edges,
        scale=scale,
        summary=width_function,
        **width_function_kwargs,
    )

    # Storage for the filtered data
    data = np.ones_like(smap.data) * fill

    # Calculate the filter value for each radial bin.
    for i in tqdm(range(radial_bin_edges.shape[1]), desc="NRGF: ", disable=not progress):
        here = np.logical_and(map_r >= radial_bin_edges[0, i], map_r < radial_bin_edges[1, i])
        here = np.logical_and(here, map_r > application_radius)
        data[here] = smap.data[here] - radial_intensity[i]
        if radial_intensity_distribution_summary[i] != 0.0:
            data[here] = data[here] / radial_intensity_distribution_summary[i]

    new_map = sunpy.map.Map(data, smap.meta)
    new_map.plot_settings["norm"] = None
    return new_map


def _set_attenuation_coefficients(order, mean_attenuation_range=None, std_attenuation_range=None, cutoff=0):
    """
    This is a helper function to Fourier Normalizing Radial Gradient Filter
    (`sunkit_image.radial.fnrgf`).

    This function sets the attenuation coefficients in the one of the following two manners:

    If ``cutoff`` is ``0``, then it will set the attenuation coefficients as linearly decreasing between
    the range ``mean_attenuation_range`` for the attenuation coefficients for mean approximation and ``std_attenuation_range`` for
    the attenuation coefficients for standard deviation approximation.

    If ``cutoff`` is not ``0``, then it will set the last ``cutoff`` number of coefficients equal to zero
    while all the others the will be set as linearly decreasing as described above.

    .. note::

        This function only describes some of the ways in which attenuation coefficients can be calculated.
        The optimal coefficients depends on the size and quality of image. There is no generalized formula
        for choosing them and its up to the user to choose a optimum value.

    .. note::

        The returned maps have their ``plot_settings`` changed to remove the extra normalization step.

    Parameters
    ----------
    order : `int`
        The order of the Fourier approximation.
    mean_attenuation_range : `list`, optional
        A list of length of ``2`` which contains the highest and lowest values between which the coefficients for
        mean approximation be calculated in a linearly decreasing manner.
    std_attenuation_range : `list`, optional
        A list of length of ``2`` which contains the highest and lowest values between which the coefficients for
        standard deviation approximation be calculated in a linearly decreasing manner.
    cutoff : `int`, optional
        The numbers of coefficients from the last that should be set to ``zero``.

    Returns
    -------
    `numpy.ndarray`
        A numpy array of shape ``[2, order + 1]`` containing the attenuation coefficients for the Fourier
        coffiecients. The first row describes the attenustion coefficients for the Fourier coefficients of
        mean approximation. The second row contains the attenuation coefficients for the Fourier coefficients
        of the standard deviation approximation.
    """
    if std_attenuation_range is None:
        std_attenuation_range = [1.0, 0.0]
    if mean_attenuation_range is None:
        mean_attenuation_range = [1.0, 0.0]
    attenuation_coefficients = np.zeros((2, order + 1))
    attenuation_coefficients[0, :] = np.linspace(mean_attenuation_range[0], mean_attenuation_range[1], order + 1)
    attenuation_coefficients[1, :] = np.linspace(std_attenuation_range[0], std_attenuation_range[1], order + 1)

    if cutoff > (order + 1):
        msg = "Cutoff cannot be greater than order + 1."
        raise ValueError(msg)

    if cutoff != 0:
        attenuation_coefficients[:, (-1 * cutoff) :] = 0

    return attenuation_coefficients


def fnrgf(
    smap,
    *,
    radial_bin_edges=None,
    order=3,
    mean_attenuation_range=None,
    std_attenuation_range=None,
    cutoff=0,
    ratio_mix=None,
    intensity_summary=np.nanmean,
    width_function=np.std,
    application_radius=1 * u.R_sun,
    number_angular_segments=130,
    progress=False,
    fill=np.nan,
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
    the entire radial bin, this Fourier approximated value is then used to normalize the intensity in the
    radial bin.

    .. note::

        The returned maps have their ``plot_settings`` changed to remove the extra normalization step.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        A SunPy map.
    radial_bin_edges : `astropy.units.Quantity`, optional
        A two-dimensional array of bin edges of size ``[2, nbins]`` where ``nbins`` is
        the number of bins.
        Defaults to `None` which will use equally spaced bins.
    order : `int`, optional
        Order (number) of fourier coefficients and it can not be lower than 1.
        Defaults to 3.
    mean_attenuation_range : `list`, optional
        A list of length of ``2`` which contains the highest and lowest values between which the coefficients for
        mean approximation be calculated in a linearly decreasing manner.
        Defaults to `None`.
    std_attenuation_range : `list`, optional
        A list of length of ``2`` which contains the highest and lowest values between which the coefficients for
        standard deviation approximation be calculated in a linearly decreasing manner.
        Defaults to `None`.
    cutoff : `int`, optional
        The numbers of coefficients from the last that should be set to ``zero``.
        Defaults to 0.
    ratio_mix : `float`, optional
        A one dimensional array of shape ``[2, 1]`` with values equal to ``[K1, K2]``.
        The ratio in which the original image and filtered image are mixed.
        Defaults to ``[15, 1]``.
    intensity_summary :``function``, optional
        A function that returns a summary statistic of the radial intensity.
        Default is `numpy.nanmean`.
    width_function : ``function``
        A function that returns a summary statistic of the distribution of intensity, at a given radius.
        Defaults to `numpy.std`.
    application_radius : `astropy.units.Quantity`
        The FNRGF is applied to emission at radii above the application_radius.
        Defaults to 1 solar radii.
    number_angular_segments : `int`
        Number of angular segments in a circular annulus.
        Defaults to 130.
    progress : `bool`, optional
        Show a progressbar while computing.
        Defaults to `False`.
    fill : Any, optional
        The value to be placed outside of the bounds of the algorithm.
        Defaults to NaN.

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

    if ratio_mix is None:
        ratio_mix = [15, 1]
    if order < 1:
        msg = "Minimum value of order is 1"
        raise ValueError(msg)

    # Handle the bin edges and radius
    radial_bin_edges, map_r = find_radial_bin_edges(smap, radial_bin_edges)

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
    data = np.ones_like(smap.data) * fill

    # Set attenuation coefficients
    attenuation_coefficients = _set_attenuation_coefficients(order, mean_attenuation_range, std_attenuation_range, cutoff)

    # Iterate over each circular ring
    for i in tqdm(range(nbins), desc="FNRGF: ", disable=not progress):
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
                ],
            ),
        )
        sin_matrix = np.sin(
            np.array(
                [
                    [(2 * np.pi * (j + 1) * (i + 0.5)) / number_angular_segments for j in range(order)]
                    for i in range(number_angular_segments)
                ],
            ),
        )

        # Iterate over each segment in a circular ring
        for j in range(number_angular_segments):
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

        # Get the approximated value of mean
        mean_approximated = np.matmul(fourier_coefficients_a_k, np.cos(angles_of_pixel))
        mean_approximated += np.matmul(fourier_coefficients_b_k, np.sin(angles_of_pixel))
        mean_approximated += fourier_coefficient_a_0 / 2

        # Get the approximated value of standard deviation
        std_approximated = np.matmul(fourier_coefficients_c_k, np.cos(angles_of_pixel))
        std_approximated += np.matmul(fourier_coefficients_d_k, np.sin(angles_of_pixel))
        std_approximated += fourier_coefficient_c_0 / 2

        # Normalize the data
        # Refer equation (7) in the paper
        std_approximated = np.where(std_approximated == 0.00, 1, std_approximated)
        data[annulus] = np.ravel((smap.data[annulus] - mean_approximated) / std_approximated)

        # Linear combination of original image and the filtered data.
        data[annulus] = ratio_mix[0] * smap.data[annulus] + ratio_mix[1] * data[annulus]

    new_map = sunpy.map.Map(data, smap.meta)
    new_map.plot_settings["norm"] = None
    return new_map


@u.quantity_input(application_radius=u.R_sun, vignette=u.R_sun)
def rhef(
    smap,
    *,
    radial_bin_edges=None,
    application_radius=0 * u.R_sun,
    upsilon=0.35,
    method="numpy",
    vignette=None,
    progress=False,
    fill=np.nan,
):
    """
    Implementation of the Radial Histogram Equalizing Filter (RHEF).

    The filter works as follows:

    Radial Histogram Equalization is a simple algorithm for removing the radial gradient to reveal
    coronal structure. It also significantly improves the visualization of high dynamic range solar imagery.
    RHE takes the input map and bins the pixels by radius, then ranks the elements in each bin sequentially and normalizes the set to 1.

    .. note::

        The returned maps have their ``plot_settings`` changed to remove the extra normalization step.

    Parameters
    ----------
    smap : `sunpy.map.Map`
        The SunPy map to enhance using the RHEF algorithm.
    radial_bin_edges : `astropy.units.Quantity`, optional
        A two-dimensional array of bin edges of size ``[2, nbins]`` where ``nbins`` is the number of bins.
        These define the radial segments where filtering is applied.
        If None, radial bins will be generated automatically.
    application_radius : `astropy.units.Quantity`, optional
        The radius above which to apply the RHEF. Only regions with radii above this value will be filtered.
        Defaults to 0 solar radii.
    upsilon : float or None, optional
        A double-sided gamma function to apply to modify the equalized histograms. Defaults to 0.35.
    method : ``{"inplace", "numpy", "scipy"}``, optional
        Method used to rank the pixels for equalization.
        Defaults to 'inplace'.
    vignette : `astropy.units.Quantity`, optional
        Radius beyond which pixels will be set to NaN.
        Must be in units that are compatible with "R_sun" as the value will be transformed.
        Defaults to `None`.
    progress : `bool`, optional
        Show a progressbar while computing.
        Defaults to `False`.
    fill : Any, optional
        The value to be placed outside of the bounds of the algorithm.
        Defaults to NaN.

    Returns
    -------
    `sunpy.map.Map`
        A SunPy map with the Radial Histogram Equalizing Filter applied to it.

    References
    ----------
    * Gilly & Cranmer 2024, in prep.

    * The implementation is highly inspired by this doctoral thesis:
      Gilly, G. Spectroscopic Analysis and Image Processing of the Optically-Thin Solar Corona
      https://www.proquest.com/docview/2759080511
    """

    radial_bin_edges, map_r = find_radial_bin_edges(smap, radial_bin_edges)

    data = np.ones_like(smap.data) * fill

    # Select the ranking method
    ranking_func = _select_rank_method(method)

    # Loop over each radial bin to apply the filter
    for i in tqdm(range(radial_bin_edges.shape[1]), desc="RHEF: ", disable=not progress):
        # Identify pixels within the current radial bin
        here = np.logical_and(
            map_r >= radial_bin_edges[0, i].to(u.R_sun), map_r < radial_bin_edges[1, i].to(u.R_sun)
        )
        if application_radius is not None and application_radius > 0:
            here = np.logical_and(here, map_r >= application_radius)
        # Apply ranking function
         
        data[here] = ranking_func(smap.data[here])
        if upsilon is not None:
            data[here] = apply_upsilon(data[here], upsilon)

    new_map = sunpy.map.Map(data, smap.meta)

    if vignette is not None:
        new_map = blackout_pixels_above_radius(new_map, vignette.to(u.R_sun))

    # Adjust plot settings to remove extra normalization
    # This must be done whenever one is adjusting
    # the overall statistical distribution of values
    new_map.plot_settings["norm"] = None

    # Return the new SunPy map with RHEF applied
    return new_map
