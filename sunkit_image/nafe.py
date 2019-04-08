"""
This module contains functions that can be used to enchance tof fine structures in extreme
ultraviolet images of the corona.
"""
import argparse
import warnings
import multiprocessing
from collections import namedtuple
from multiprocessing import Pool

import tqdm
import numpy as np
import skimage.filters

import sunpy.map

__all__ = [
    "nafe"
]


_PREDEF_PARAMS = {
    # defined by original program
    # wavelength    # gamma     # nafe
    "4500":          (1.0,        .05),
    "1700":          (2.2,        .10),
    "1600":          (2.2,        .10),
    "335":           (3.0,        .20),
    "304":           (5.5,        .15),
    "211":           (2.7,        .20),
    "193":           (2.6,        .20),
    "171":           (2.6,        .20),
    "131":           (2.9,        .20),
    "94":            (3.1,        .20),
    "def":           (2.2,        .20)
}

_SharedParams = namedtuple("_SharedParams", ["image", "membership", "hist_bins", "gamma", "sigma",
                                             "weight", "in_margin", "halfn", "out_margin"])
_SHARED_PARAMS = None  # for sharing the image and parameters across processes


def nafe(input_map, in_margin=None, out_margin=(0, 2 ** 16 - 1), gamma=None, nafe_weight=None,
         hist_bins= 2 ** 8 - 1, noise_reduction_sigma=15, n=129, c=np.ones(12), nproc=None):
    """
    Enhances the input map via "Noise Adaptive Fuzzy Equalization". Ideally suited for the
    visualization of fine structures in extreme ultraviolet images of the corona. Particularly
    suited for the exceptionally high dynamic range images from the Atmospheric Imaging Assembly
    instrument on the Solar Dynamics Observatory. This method produces artifact-free images and
    gives significantly better results than methods based on convolution or Fourier transform which
    are often used for that purpose.

    Parameters
    ----------
    input_map: `~sunpy.map.Map`
        The original input map to be enhanced.
    in_margin: `tuple`
        (low, high) input image values. Pixels with values < low or > high are set to  0.
    out_margin: `tuple`
        (low, high) output image values. Deaults to (0, 2 ** 16 - 1).
    gamma: `float`
        The gamma value used for gamma_transform.
    nafe_weight: `float`
        Controls the ratio of the nafe component in the result image. Typical values for nafe_weight
        lie in the interval [0.05, 0.3].
    hist_bins: `int`
        Number of histogram bins used for calculating fuzzy historgam for nafe component. Defaults
        to 2 ** 8 - 1.
    noise_reduction_sigma: `float`
        Sigma of artificial gaussian noise added to the input image for nafe component. Defaults to
        15.
    n: `int`
        Full kernel width used for calculating neighborhood for nafe component. Must be an odd
        integer. Defaults to 129.
    c: `~numpy.ndarray`
        Fuzzy neighborhood constants for nafe component. Defaults to 12 layers each of coefficient
        1.
    nproc : `int`
        Number of processes for multiprocessing. Defaults to number of available cpu cores.

    Returns
    -------
    enhanced_map: `~sunpy.map.Map`
        The enhanced map resulting from the linear combination of two images,
        gamma_transform(input_map) and nafe_transform(input_map) such that:

            enhanced_map = (1 - nafe_weight) * gamma_component + nafe_weight * nafe_component

    References
    ----------

    * `A Noise Adaptive Fuzzy Equalization Method for Processing Solar Extreme Ultraviolet Images <https://doi.org/10.1023/A:1005226402796>`__

    """
    global _SHARED_PARAMS
    data = input_map.data
    metadata = input_map.meta
    predef_gamma, predef_weight = _PREDEF_PARAMS.get(metadata["wavelnth"], _PREDEF_PARAMS["def"])

    if in_margin is None:
        in_margin = (data.min(), data.max())
    else:
        data = data.clip(min=in_margin[0], max=in_margin[1])

    if gamma is None:
        gamma = predef_gamma
    elif gamma < 0:
        raise ValueError("gamma should be a non-negative real number! gamma given is:", gamma)

    if nafe_weight == 0:
        warnings.warn("nafe_weight is set to zero which disables nafe", RuntimeWarning)
        enhanced_image = _transform(in_margin=in_margin, out_margin=out_margin, old_value=data,
                                    power=gamma)
    else:
        if nafe_weight is None:
            nafe_weight = predef_weight
        elif nafe_weight < 0 or nafe_weight > 1:
            raise ValueError("nafe_weight should be between 0 and 1. nafe_weight given is:",
                             nafe_weight)
        elif nafe_weight < 0.05 or nafe_weight > 0.3:
            warnings.warn("typical values for nafe_weight should be between 0.05 and 0.3! "
                          "nafe_weight given is: " + str(nafe_weight), RuntimeWarning)

        if not isinstance(hist_bins, int) or hist_bins <= 0:
            raise ValueError("hist_bins should be a positive integer number! hist_bins given is:",
                             hist_bins)

        if noise_reduction_sigma < 0.0:
            raise ValueError("noise_reduction_sigma should be a non-negative real number! "
                             "noise_reduction_sigma given is:", noise_reduction_sigma)

        if n <= 0 or n % 2 == 0 or not isinstance(n, int):
            raise ValueError("n should be an odd positive integer number! n given is:", n)

        if nproc is None:
            nproc = multiprocessing.cpu_count()
        elif not isinstance(nproc, int) or nproc <= 0:
            raise ValueError("nproc should be a non-negative integer number! nproc given is:",
                             nproc)
        elif nproc > multiprocessing.cpu_count():
            raise ValueError("nproc given is too big! Maximum value should be:",
                             multiprocessing.cpu_count())

        # Shared nafe parameters
        membership = _get_membership_grade(n=n, wl=c)
        halfn = n // 2
        _SHARED_PARAMS = _SharedParams(image=data, membership=membership, hist_bins=hist_bins,
                                       gamma=gamma, sigma=noise_reduction_sigma, weight=nafe_weight,
                                       in_margin=in_margin, halfn=halfn,
                                       out_margin=(0, 2 ** 16 - 1))

        # NAFE multiprocessing context
        nrows, ncols = data.shape
        size = nrows * ncols
        rows, cols = np.meshgrid(range(nrows), range(ncols), indexing='ij')
        indices = zip(rows.flat, cols.flat)
        with Pool(processes=nproc) as pool:
            # return margin [0, 2 ** 16 - 1]
            nafe_image = list(tqdm.tqdm(pool.imap(_nafe_slice, indices), total=size))

        enhanced_image = np.array(nafe_image).reshape(nrows, ncols)
        enhanced_image = _transform(in_margin=(0., 2. ** 16 - 1.), out_margin=out_margin,
                                    old_value=enhanced_image)

    return sunpy.map.Map((enhanced_image, metadata))


def _get_membership_grade(n=129, wl=np.ones(12)):
    """
    Calculates the matrix used for expressing the membership grade of neighborhood pixels to the
    fuzzy histogram computed for each pixel in image. The matrix L is a linear combination of
    M Gaussian kernels calculated as:

        L(x, y) = d * sum[m=1:M](wl[m] * G(sigma[m], x, y))

    where:
    x = horizontal distance from center
    y = vertical distance from center
    sigma[m] = 2 ** (m / 2)
    M = len(wl)

    d = 2 * pi / sum[m=1:M]( wl[m] * (sigma[m] ** - (1/2)) )
    G(s, x, y) = exp(−0.5 * (x ** 2) + (y ** 2)) * (s ** -2)) / (2 * pi * sqrt(s)), s = sigma[m]

    Parameters
    ----------
    n: `int`
        Full neighborhood width. Must be an odd integer and defaults to 129.
    wl: `~numpy.ndarray`
        Weights list of for each Gaussian kernel of the M Gaussian kernels
        used for computing L. M is calculated as len(wl). Defaults to a
        1x12 matrix of ones.

    Returns
    -------
    L: `~numpy.ndarray`
        Membership grade matrix.

    Example
    -------
    >>> _get_membership_grade(5)
    array([[0.65289406, 0.73715583, 0.7741782 , 0.73715583, 0.65289406],
           [0.73715583, 0.86813217, 0.92826105, 0.86813217, 0.73715583],
           [0.7741782 , 0.92826105, 1.        , 0.92826105, 0.7741782 ],
           [0.73715583, 0.86813217, 0.92826105, 0.86813217, 0.73715583],
           [0.65289406, 0.73715583, 0.7741782 , 0.73715583, 0.65289406]])
    """
    M = len(wl)
    sigma = [2 ** (m / 2) for m in range(1, M+1)]  # starts from 1 for maths
    d = 2 * np.pi / np.sum(wl * np.power(sigma, -1/2))

    wlG = 0
    for s, w in zip(sigma, wl):
        G = lambda r, c: np.exp(
                -0.5 * ((r - n//2) ** 2 + (c - n//2) ** 2) * (s ** -2)) / (2 * np.pi * np.sqrt(s))
        wlG += w * np.fromfunction(G, (n, n))

    return d * wlG


def _transform(in_margin, out_margin, old_value, power=1):
    """
    Gamma + Linear transform
    """
    in_low = in_margin[0]
    out_low = out_margin[0]
    in_range = in_margin[1] - in_margin[0]
    out_range = out_margin[1] - out_margin[0]
    new_value = (old_value - in_low) / in_range

    return out_low + out_range * (new_value ** (1 / power))


def _nafe_slice(index):
    """
    Core NAFE function running on one pixel for multiprocessing
    """
    global _SHARED_PARAMS
    image = _SHARED_PARAMS.image
    membership = _SHARED_PARAMS.membership
    hist_bins = _SHARED_PARAMS.hist_bins
    gamma = _SHARED_PARAMS.gamma
    sigma = _SHARED_PARAMS.sigma
    weight = _SHARED_PARAMS.weight
    in_margin = _SHARED_PARAMS.in_margin
    halfn = _SHARED_PARAMS.halfn
    out_margin = _SHARED_PARAMS.out_margin

    row, col = index
    rows, cols = image.shape

    img_minrow = max(0, row - halfn)
    img_mincol = max(0, col - halfn)
    img_maxrow = min(rows, row + halfn + 1)
    img_maxcol = min(cols, col + halfn + 1)

    mem_minrow = img_minrow - row + halfn
    mem_mincol = img_mincol - col + halfn
    mem_maxrow = img_maxrow - row + halfn
    mem_maxcol = img_maxcol - col + halfn

    neighborhood = image[img_minrow: img_maxrow, img_mincol: img_maxcol]
    membership = membership[mem_minrow: mem_maxrow, mem_mincol: mem_maxcol]

    hist, bins = np.histogram(neighborhood, bins=hist_bins, weights=membership)
    sigma = sigma / ((bins[-1] - bins[0]) / hist_bins)
    bins = (bins[:-1] + bins[1:]) / 2

    cumsum_hist = np.cumsum(hist)
    cumsum_hist /= cumsum_hist[-1]  # normalize
    # add artificial noise, reshaping to dummy 2d to enable 2d gauss from skimage then
    # flattening again
    cum_fuzzy_hist = skimage.filters.gaussian(cumsum_hist.reshape(hist_bins, 1), sigma=sigma)
    cum_fuzzy_hist = cum_fuzzy_hist.flat

    old_val = image[row, col]
    gamma_val = _transform(in_margin=in_margin, out_margin=out_margin, old_value=old_val,
                           power=gamma)
    nafe_val = np.interp(image[row, col], bins, cum_fuzzy_hist)  # returns [0., 1.]
    nafe_val = _transform(in_margin=(0., 1.), out_margin=out_margin, old_value=nafe_val)

    return (1 - weight) * gamma_val + weight * nafe_val
