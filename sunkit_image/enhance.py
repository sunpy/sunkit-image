"""
This module contains enhancement routines for solar physics data.
"""

import warnings

import numpy as np
import scipy.ndimage as ndimage

from sunkit_image.utils.decorators import accept_array_or_map

__all__ = ["mgn", "wow"]


@accept_array_or_map(arg_name="data")
def mgn(
    data,
    *,
    sigma=None,
    k=0.7,
    gamma=3.2,
    h=0.7,
    weights=None,
    truncate=3,
    clip=True,
    gamma_min=None,
    gamma_max=None,
):
    """
    Multi-scale Gaussian normalization.

    This function can be used to visualize information over a wide range of spatial scales. It
    works by normalizing the image by calculating local mean and standard deviation over many
    spatial scales by convolving with Gaussian kernels of different standard deviations. All the
    normalized images are then arctan transformed (similar to a gamma transform). Then all the
    images are combined by adding all of them after multiplying with suitable weights. This method
    can be used to reveal information and structures at various spatial scales.

    .. note::

        * In practice, the weights and h may be adjusted according to the desired output, and also according
          to the type of input image (e.g. wavelength or channel). For most purposes, the weights can be set
          equal for all scales.
        * We do not deal with NaN (Not a Number) in this implementation.
        * The input data array should be normalized by the exposure time.
        * The input data array should be dtype `float`.

    Parameters
    ----------
    data : `numpy.ndarray`, `sunpy.map.GenericMap`
        Image to be transformed.
    sigma : `list` of `float`, optional
        Range of Gaussian widths (i.e. the standard deviation of the Gaussian kernel) to transform over.
        Defaults to ``[1.25, 2.5, 5, 10, 20, 40]``.
    k : `float`, optional
        The scaling factor multiplied with each Gaussian transformed image before applying the arctan transformation.
        Essentially controls the severity of the arctan transformation.
        Defaults to 0.7.
    gamma : `float`, optional
        The value used to calculate the global gamma-transformed image.
        Ideally should be between 2.5 to 4 according to the paper.
        Defaults to 3.2.
    gamma_min : `float`, optional
        Minimum input to the gamma transform.
        Defaults to minimum value of ``data``.
    gamma_max : `float`, optional
        Maximum input to the gamma transform.
        Defaults to maximum value of ``data``.
    h : `float`, optional
        Weight of global filter to Gaussian filters.
        Defaults to 0.7.
    weights : `list`, optional
        Used to weight all the transformed images during the calculation of the final image.
        Defaults to all weights are one.
    truncate : `int`, optional
        The number of sigmas to truncate the kernel.
        Defaults to 3.
    clip : `bool`, optional
        If set to `True` it will clip all the non-positive values in the image to a very small positive
        value. Defaults to `True`.

    Returns
    -------
    `numpy.ndarray` or `sunpy.map.GenericMap`
        Normalized image.
        If a map is input, a map is returned with new data and the same metadata.

    References
    ----------
    * Huw Morgan and Miloslav Druckmüller.
      "Multi-scale Gaussian normalization for solar image processing."
      Sol Phys 289, 2945-2955, 2014
      `doi:10.1007/s11207-014-0523-9 <https://doi.org/10.1007/s11207-014-0523-9>`__
    """
    olderr = np.seterr(all='ignore')
    if sigma is None:
        sigma = [1.25, 2.5, 5, 10, 20, 40]
    if np.isnan(data).any():
        warnings.warn(
            "One or more entries in the input data are NaN. This implementation does not account "
            "for the presence of NaNs in the input data. As such, this may result in undefined "
            "behavior.",
            stacklevel=3,
        )
    if weights is None:
        weights = np.ones(len(sigma))
    # 1. Replace spurious negative pixels with zero
    if clip is True:
        data[data <= 0] = 1e-15
    # We want to avoid casting issues between dtype('float64') and dtype('int16')
    data = data.astype(np.float32)
    image = np.zeros_like(data)
    conv = np.zeros_like(data)
    sigmaw = np.zeros_like(data)
    for s, weight in zip(sigma, weights, strict=True):
        # 2 & 3 Create kernel and convolve with image
        # Refer to equation (1) in the paper
        ndimage.gaussian_filter(data, sigma=s, truncate=truncate, mode="nearest", output=conv)
        # 4. Calculate difference between image and the local mean image,
        # square the difference, and convolve with kernel. Square-root the
        # resulting image to give `local standard deviation` image sigmaw
        # Refer to equation (2) in the paper
        conv = data - conv
        ndimage.gaussian_filter(conv**2, sigma=s, truncate=truncate, mode="nearest", output=sigmaw)
        np.sqrt(sigmaw, out=sigmaw)
        # 5. Normalize the gaussian transformed image to give C_i.
        sigmaw = np.where(sigmaw == 0.0, 1.0, sigmaw)
        conv /= sigmaw
        # 6. Apply arctan transformation on Ci to give C'i
        # Refer to equation (3) in the paper
        conv *= k
        np.arctan(conv, out=conv)
        conv *= weight
        image += conv
    # Delete these arrays here as it reduces the total memory consumption when
    # we create the Cprime_g temp array below.
    del conv
    del sigmaw
    # 8. Take weighted mean of C'i to give a weighted mean locally normalised image.
    image /= len(sigma)
    # 9. Calculate global gamma-transformed image C'g
    # Refer to equation (4) in the paper
    gamma_min = data.min() if gamma_min is None else gamma_min
    gamma_max = data.max() if gamma_max is None else gamma_max
    Cprime_g = data - gamma_min
    if (gamma_max - gamma_min) != 0.0:
        Cprime_g /= gamma_max - gamma_min
    Cprime_g **= 1 / gamma
    Cprime_g *= h
    # 10. Sum the weighted mean locally transformed image with the global normalized image
    # Refer to equation (5) in the paper
    image *= 1 - h
    image += Cprime_g
    np.seterr(**olderr)
    return image


@accept_array_or_map(arg_name="data")
def wow(
    data,
    *,
    scaling_function=None,
    n_scales=None,
    weights=None,
    whitening=True,
    denoise_coefficients=None,
    noise=None,
    bilateral=None,
    bilateral_scaling=False,
    soft_threshold=True,
    preserve_variance=False,
    gamma=3.2,
    gamma_min=None,
    gamma_max=None,
    h=0,
):
    """
    Processes an image with the Wavelets Optimized Whitening (WOW) algorithm.

    This function manipulates the wavelet spectrum of the input image so that the power in the output image
    is equal at all locations and all spatial scales, thus "whitening" the wavelet spectrum. By doing so, the
    large scale structures are attenuated and the small scale structures are reinforced, with the objective
    criterion that the variance at all scales must be equal. The algorithm has the added advantage to allow
    attenuation of the noise at the same time, thus avoiding the usual explosion of noise usually inherent to
    enhancement methods.

    .. note::

        * The filter was developed with the intention to find an objective criterion for enhancement.
          By default, there are therefore no free parameters to adjust, except for the denoising
          coefficients. It is however possible to set the synthesis weights to be values other than 1 to tune
          the output. It is also possible to merge the output with a gamma-scaled version of the original
          image. The weight of the gamma scaled image can be adjusted using
          the parameter ``h``, to the type of input image (e.g., wavelength or channel).
          For most purposes, the weights can be set to be equal for all scales.
        * We do not deal with NaN (Not a Number) in this implementation.

    Parameters
    ----------
    data : `numpy.ndarray` or `sunpy.map.GenericMap`
        Image to be transformed.
    scaling_function : {``Triangle`` , ``B3spline``}, optional
        The wavelet scaling function, comes from the ``watroo`` package.
        Defaults to ``B3spline``.
    n_scales : `int`, optional
        Number of scales used for the wavelet transform.
        If `None`, the number of scales is computed to be the maximum compatible with the size of the input.
        Defaults to None.
    weights : `list` of `float`, optional
        Optional reconstruction weights used in the synthesis stage.
        By default, the weights are all set to 1.
        If the weights are not 1, the spectrum of the output is not white.
        Defaults to ``[]``.
    whitening : `bool`
        If True (default), the spectrum is whitened, i.e., normalized to the local power at each scale.
        Defaults to `True`.
    denoise_coefficients : `list` of `float`, optional
        Noise threshold, in units of the noise standard deviation, used at each scale to denoise the wavelet
        coefficients.
        Defaults to ``[]``.
    noise : `numpy.ndarray` or `None`, optional
        A map of the noise standard deviation, of same shape as the input data.
        This can be used to take into account spatially dependent (i.e., Poisson) noise.
        Defaults to `None`.
    bilateral : `int` or `None`, optional
        Uses bilateral convolution to form an edge-aware wavelet transform.
        The bilateral transform avoids the formation of glows near strong gradients.
        The recommended "natural" value is 1.
        Defaults to `None`.
    bilateral_scaling : `bool`, optional
        Experimental, do not use.
        Defaults to `False`
    soft_threshold: `bool`, optional
        Used only if denoise_coefficients is not ``[]``.
        If `True`, soft thresholding is used for denoising, otherwise, hard thresholding is used.
        Soft thresholding tends to create less artifacts.
        Defaults to `True`.
    preserve_variance: `bool`, optional
        Experimental, do not use.
        Defaults to `False`.
    gamma: `float`, optional
        The value used to calculate the global gamma-transformed image.
        Defaults to 3.2.
    gamma_min : `float` or `None`, optional
        Minimum input to the gamma transform.
        If `None`, defaults to minimum value of ``data``.
    gamma_max : `float` or None, optional
        Maximum input to the gamma transform.
        If None, defaults to maximum value of ``data``.
    h : `float`, optional
        Weight of the gamma-scaled image wrt that of the filtered image.
        Defaults to 0.

    Returns
    -------
    `numpy.ndarray` or `sunpy.map.GenericMap`
        Normalized image.
        If a map is input, a map is returned with new data and the same metadata.

    References
    ----------
    * Frédéric Auchère, Elie Soubrié, Gabriel Pelouze, Eric Buchlin, 2023,
      "Image Enhancement with Wavelets Optimized Whitening.", Astronomy & Astrophysics, 670, A66
      `doi:10.1051/0004-6361/202245345 <https://doi.org/10.1051/0004-6361/202245345>`__
    """
    try:
        from watroo import B3spline, utils
    except ImportError as e:
        msg = "The `watroo` package is required to use the `wow` function. Please install it first."
        raise ImportError(msg) from e

    if denoise_coefficients is None:
        denoise_coefficients = []
    if weights is None:
        weights = []
    if scaling_function is None:
        scaling_function = B3spline
    wow_image, _ = utils.wow(
        data,
        scaling_function=scaling_function,
        n_scales=n_scales,
        weights=weights,
        whitening=whitening,
        denoise_coefficients=denoise_coefficients,
        noise=noise,
        bilateral=bilateral,
        bilateral_scaling=bilateral_scaling,
        soft_threshold=soft_threshold,
        preserve_variance=preserve_variance,
        gamma=gamma,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        h=h,
    )
    return wow_image
