"""
This module contains functions that will enhance the entire image.
"""

import numpy as np
import scipy.ndimage as ndimage

__all__ = [
    "mgn",
]


def mgn(
    data,
    sigma=[1.25, 2.5, 5, 10, 20, 40],
    k=0.7,
    gamma=3.2,
    h=0.7,
    weights=None,
    truncate=3
):

    """
    Multi-scale Gaussian normalization.

    Extreme ultra-violet images of the corona contain information over a wide range of spatial scales,
    and different structures such as active regions, quiet Sun, and filament channels contain information
    at very different brightness regimes. The MGN method normalises an image by using the local mean and
    standard deviation calculated using a Gaussian-weighted sample of local pixels. This normalised image
    is transformed by the arctan function (similar to a gamma transformation). This is applied over
    several spatial scales, and the final image is a weighted combination of the normalised components.
    The method reveals information at the finest scales whilst maintaining enough of the larger-scale
    information to provide context. It also intrinsically flattens noisy regions and can reveal structure
    in off-limb regions out to the edge of the field of view.

    .. note::
        In practice, the weights and h may be adjusted according to the desired output, and also according
        to the type of input image
        (e.g. wavelength or channel).
        For most purposes, the weights can be set
        equal for all scales.

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
        Defaults to 0.7
    gamma : `float`, optional
        The value used to calculate the  global gamma-transformed image.
        Ideally should be between 2.5 to 4 according to the paper.
        Defaults to 3.2
    h : `float`, optional
        Weight of global filter to Gaussian filters.
        Defaults to 0.7
    weights : `list`, optional
        Used to weight all the transformed images during the calculation of the
        final image. If not specificed, all weights are one.
    truncate : `int`, optional 
        The number of sigmas (defaults to 3) to truncate the kernel.

    Returns
    -------
    `numpy.ndarray`
        Normalized image.

    References
    ----------
    * Morgan, Huw, and Miloslav Druckmuller.
      "Multi-scale Gaussian normalization for solar image processing."
      arXiv preprint arXiv:1403.6613 (2014).
      Ref: Sol Phys (2014) 289: 2945. doi:10.1007/s11207-014-0523-9
    """

    if weights is None:
        weights = np.ones(len(sigma))

    # 1. Replace spurious negative pixels with zero
    data[data <= 0] = 1e-15  # Makes sure that all values are above zero
    image = np.zeros_like(data)
    conv = np.zeros_like(data)
    sigmaw = np.zeros_like(data)

    for s, weight in zip(sigma, weights):
        # 2 & 3 Create kernel and convolve with image
        # Refer to equation (1) in the paper
        ndimage.filters.gaussian_filter(data, sigma=s,
                                        truncate=truncate, mode='nearest', output=conv)

        # 4. Calculate difference between image and the local mean image,
        # square the difference, and convolve with kernel. Square-root the
        # resulting image to give ‘local standard deviation’ image sigmaw
        # Refer to equation (2) in the paper
        conv = data - conv
        ndimage.filters.gaussian_filter(conv ** 2, sigma=s,
                                        truncate=truncate, mode='nearest', output=sigmaw)
        np.sqrt(sigmaw, out=sigmaw)
        
        # 5. Normalize the gaussian transformed image to give C_i.
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

    # 8. Take weighted mean of C'i to give a weighted mean locally normalised
    # image.
    image /= len(sigma)

    # 9. Calculate global gamma-transformed image C'g
    # Refer to equation (4) in the paper
    data_min = data.min()
    data_max = data.max()
    Cprime_g = (data - data_min)
    Cprime_g /= (data_max - data_min)
    Cprime_g **= (1/gamma)
    Cprime_g *= h

    # 10. Sum the weighted mean locally transformed image with the global normalized image
    # Refer to equation (5) in the paper
    image *= (1 - h)
    image += Cprime_g

    return image
