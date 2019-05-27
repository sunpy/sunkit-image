"""
This module contains functions that can be used to enhance the entire solar image.
"""

import numpy as np
import scipy.ndimage as ndimage

__all__ = [
    "mgn1"
]

def mgn1(
    data,
    sigma = [1.25, 2.5, 5, 10, 20, 40],
    k = 0.7,
    gamma=3.2,
    h=0.7,
    weights=None,
    truncate = 3
):
    
    """
    Multi-scale Gaussian normalization.
    
    Parameters
    ----------
    data : `numpy.ndarray`
        Image to be transformed.
    sigma : `list`, optional
        Range of guassian widths to transform over.
    k : `float`, optional
        Controls the severity of the arctan transformation.
        Defaults to 0.7
    gamma : `float`, optional
        The value used to calulcate the  global gamma-transformed image.
        Ideally should be between 2.5 to 4.
        Defaults to 3.2
    h : `float`, optional
        Weight of global filter to gaussian filters.
        Defaults to 0.7
    weights : `list`, optional
        Used to weight all the transformed images during the calculation of the
        final image. If not specificed, all weights are one.
    width : `int`
        An odd integer defining the width of the kernel to be convolved.
    truncate : `int`
        The number of sigmas to truncate the kernel.
        Defaults to 3
    
    Returns
    -------
    image: `numpy.ndarray`
        Normalized image.
    
    Reference
    ---------
    * Morgan, Huw, and Miloslav Druckmuller. "Multi-scale Gaussian normalization for solar image processing."
    arXiv preprint arXiv:1403.6613 (2014).
    Ref: Sol Phys (2014) 289: 2945. doi:10.1007/s11207-014-0523-9
    
    .. notes::
        In practice, the weights and h may be adjusted according to the desired
        output, and also according to the type of input image
        (e.g. wavelength or channel).
        For most purposes, the weights can be set
        equal for all scales.
    """
    
    #This is rectified version of Stuart's Code.
    if weights is None:
        weights = np.ones(len(sigma))

    # 1. Replace spurious negative pixels with zero
    data[data <= 0] = 1e-15  # Makes sure that all values are above zero
    image = np.empty(data.shape, dtype=data.dtype)
    conv = np.empty(data.shape, dtype=data.dtype)
    sigmaw = np.empty(data.shape, dtype=data.dtype)

    for s, weight in zip(sigma, weights):
        # 2 & 3 Create kernel and convolve with image
        ndimage.filters.gaussian_filter(data, sigma=s,
                                        truncate=truncate, mode='nearest', output=conv)
        # 5. Calculate difference between image and the local mean image,
        # square the difference, and convolve with kernel. Square-root the
        # resulting image to give ‘local standard deviation’ image sigmaw
        conv = data - conv
        ndimage.filters.gaussian_filter(conv ** 2, sigma=s,
                                        truncate=truncate, mode='nearest', output=sigmaw)
        np.sqrt(sigmaw, out=sigmaw)
        conv /= sigmaw

        # 6. Apply arctan transformation on Ci to give C'i
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
    data_min = data.min()
    data_max = data.max()
    Cprime_g = (data - data_min)
    Cprime_g /= (data_max - data_min)
    Cprime_g **= (1/gamma)
    Cprime_g *= h

    image *= (1 - h)
    image += Cprime_g

    return image

def mgn2(image, a=(5., 5000.), b=(0., 1.), w=0.3, gamma=3.2, sigma=[2.5, 5, 10, 20, 40], k=0.7):
    """
    Multi-scale Gaussian Normalization
  
    Parameters
    ----------
    image: `numpy.ndarray`
        Image to be transformed.
    a: `tuple`, optional
        Minimum and maximum input values in image. Here it is assumed to be `(5.,5000.)`
        According to the paper it should be calculated
    b: `tuple`, optional
        Minimum and maximum output values in image ([a[0], a[1]] will be scaled to [b[0], b[1]])
        Defaults to `(0., 1)`
    w: `float`, optional
        Weight of the MGN-processed image in output image.
        Defaults to 0.3
    gamma : `float`, optional
        The value used to calulcate the  global gamma-transformed image.
        Ideally should be between 2.5 to 4.
        Defaults to 3.2
    sigma : `list`, optional
        Range of guassian widths to transform over.

     Returns
    -------
    image: `numpy.ndarray`
        Normalized image.
    
    Reference
    ---------
    * Morgan, Huw, and Miloslav Druckmuller. "Multi-scale Gaussian normalization for solar image processing."
    arXiv preprint arXiv:1403.6613 (2014).
    Ref: Sol Phys (2014) 289: 2945. doi:10.1007/s11207-014-0523-9
    """

    ax, ay = image.shape
    # normalize [a[0], a[1]] input intensities to [0,1]
    image = (image.clip (a[0], a[1]) - a[0]) / (a[1] - a[0])
    imi = np.zeros_like (image)
    for s in sigma:
    # B convolved by k_w
        bwi = ndimage.gaussian_filter (image, s, mode='nearest')
        # sigma_w
        swi = np.sqrt (ndimage.gaussian_filter ((image - bwi) ** 2, s, mode='nearest'))
        # intermediate sum of C'_i
        imi += np.arctan (k * (image - bwi) / swi)
    # weighted sum of gamma-transformed input image and normalized average of C'_i,
    # normalized to [b[0], b[1]]
    return b[0] + (b[1] - b[0]) * ((1. - w) * image ** (1. / gamma)
                                    + w * (.5 + imi / (len (sigma) * np.pi)))
