"""
This module contains functions that will enhance the entire image.
"""
import warnings

import numpy as np
import scipy.ndimage as ndimage

__all__ = ["mgn"]


def mgn(
    data,
    sigma=[1.25, 2.5, 5, 10, 20, 40],
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
        * We don't deal with nan (Not A Number) in this implementation.

    Parameters
    ----------
    data : `numpy.ndarray`
        Image to be transformed.
    sigma : `list`, optional
        Range of Gaussian widths (i.e. the standard deviation of the Gaussian kernel) to transform over.
        Defaults to ``[1.25, 2.5, 5, 10, 20, 40]``.
    k : `float`, optional
        Controls the severity of the arctan transformation. The scaling factor multiplied with each
        Gaussian transformed image before applying the arctan transformation.
        Defaults to 0.7.
    gamma : `float`, optional
        The value used to calculate the global gamma-transformed image.
        Ideally should be between 2.5 to 4 according to the paper.
        Defaults to 3.2.
    gamma_min : `float`, optional
        Minimum input to the gamma transform. Defaults to minimum value of `data`
    gamma_max : `float`, optional
        Maximum input to the gamma transform. Defaults to maximum value of `data`
    h : `float`, optional
        Weight of global filter to Gaussian filters.
        Defaults to 0.7.
    weights : `list`, optional
        Used to weight all the transformed images during the calculation of the
        final image. If not specified, all weights are one.
    truncate : `int`, optional
        The number of sigmas (defaults to 3) to truncate the kernel.
    clip : `bool`, optional
        If set to `True` it will clip all the non-positive values in the image to a very small positive
        value. Defaults to `True`.

    Returns
    -------
    `numpy.ndarray`
        Normalized image.

    References
    ----------
    * Huw Morgan and Miloslav Druckmüller.
      "Multi-scale Gaussian normalization for solar image processing."
      arXiv preprint arXiv:1403.6613 (2014).
      Ref: Sol Phys (2014) 289: 2945. doi:10.1007/s11207-014-0523-9
    """
    if np.isnan(data).any():
        warnings.warn(
            "One or more entries in the input data are NaN. This implementation does not account "
            "for the presence of NaNs in the input data. As such, this may result in undefined "
            "behavior."
        )

    if weights is None:
        weights = np.ones(len(sigma))

    # 1. Replace spurious negative pixels with zero
    if clip is True:
        data[data <= 0] = 1e-15  # Makes sure that all values are above zero

    image = np.zeros_like(data)
    conv = np.zeros_like(data)
    sigmaw = np.zeros_like(data)

    for s, weight in zip(sigma, weights):
        # 2 & 3 Create kernel and convolve with image
        # Refer to equation (1) in the paper
        ndimage.filters.gaussian_filter(data, sigma=s, truncate=truncate, mode="nearest", output=conv)

        # 4. Calculate difference between image and the local mean image,
        # square the difference, and convolve with kernel. Square-root the
        # resulting image to give ‘local standard deviation’ image sigmaw
        # Refer to equation (2) in the paper
        conv = data - conv
        ndimage.filters.gaussian_filter(conv**2, sigma=s, truncate=truncate, mode="nearest", output=sigmaw)
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

    # delete these arrays here as it reduces the total memory consumption when
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

    return image
