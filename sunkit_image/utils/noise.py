"""
This module implements a series of functions for noise level estimation.
"""

import numpy as np
from scipy.ndimage import correlate
from scipy.stats import gamma
from skimage.util import view_as_windows

__all__ = ["noise_estimation", "noiselevel", "conv2d_matrix", "weak_texture_mask"]


def noise_estimation(img, patchsize=7, decim=0, confidence=1 - 1e-6, iterations=3):
    """
    Estimates the noise level of an image.

    Additive white Gaussian noise (AWGN) is a basic noise model used in Information Theory
    to mimic the effect of many random processes that occur in nature.

    Parameters
    ----------
    img: `numpy.ndarray`
        Single Numpy image array.
    patchsize : `int`, optional
        Patch size, defaults to 7.
    decim : `int`, optional
        Decimation factor, defaults to 0.
        If you use large number, the calculation will be accelerated.
    confidence : `float`, optional
        Confidence interval to determine the threshold for the weak texture.
        In this algorithm, this value is usually set the value very close to one.
        Defaults to 0.99.
    iterations : `int`, optional
        Number of iterations,  defaults to 3.

    Returns
    -------
    `dict`
        A dictionary containing the estimated noise levels, `nlevel`;  threshold to extract weak texture
        patches at the last iteration, `thresh`; number of extracted weak texture patches `num` and the
        weak texture mask, `mask`.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> noisy_image_array = np.random.randn(100, 100)
    >>> estimate = noise_estimation(noisy_image_array, patchsize=11, iterations=10)
    >>> estimate['mask'] # Prints mask
    array([[1., 1., 1., ..., 1., 1., 0.],
        [1., 1., 1., ..., 1., 1., 0.],
        [1., 1., 1., ..., 1., 1., 0.],
        ...,
        [1., 1., 1., ..., 1., 1., 0.],
        [1., 1., 1., ..., 1., 1., 0.],
        [0., 0., 0., ..., 0., 0., 0.]])
    >>> estimate['nlevel'] # Prints nlevel
    array([1.0014616])
    >>> estimate['thresh'] # Prints thresh
    array([173.61530607])
    >>> estimate['num'] # Prints num
     array([8100.])

    References
    ----------
    * Xinhao Liu, Masayuki Tanaka and Masatoshi Okutomi
      Noise Level Estimation Using Weak Textured Patches of a Single Noisy Image
      IEEE International Conference on Image Processing (ICIP), 2012.
      DOI: 10.1109/ICIP.2012.6466947

    * Xinhao Liu, Masayuki Tanaka and Masatoshi Okutomi
      Single-Image Noise Level Estimation for Blind Denoising Noisy Image
      IEEE Transactions on Image Processing, Vol.22, No.12, pp.5226-5237, December, 2013.
      DOI: 10.1109/TIP.2013.2283400
    """

    try:
        img = np.array(img)
    except:
        raise TypeError("Input image should be a NumPy ndarray")

    try:
        patchsize = int(patchsize)
    except ValueError:
        raise TypeError("patchsize must be an integer, or int-compatible, variable")

    try:
        decim = int(decim)
    except ValueError:
        raise TypeError("decim must be an integer, or int-compatible, variable")

    try:
        confidence = float(confidence)
    except ValueError:
        raise TypeError("confidence must be a float, or float-compatible, value between 0 and 1")

    if not (confidence >= 0 and confidence <= 1):
        raise ValueError("confidence must be defined in the interval 0 <= confidence <= 1")

    try:
        iterations = int(iterations)
    except ValueError:
        raise TypeError("iterations must be an integer, or int-compatible, variable")

    output = {}
    nlevel, thresh, num = noiselevel(img, patchsize, decim, confidence, iterations)
    mask = weak_texture_mask(img, patchsize, thresh)

    output["nlevel"] = nlevel
    output["thresh"] = thresh
    output["num"] = num
    output["mask"] = mask

    return output


def noiselevel(img, patchsize, decim, confidence, iterations):
    """
    Calculates the noise level of the input array.

    Parameters
    ----------
    img: `numpy.ndarray`
        Single Numpy image array.
    patchsize : `int`, optional
        Patch size, defaults to 7.
    decim : `int`, optional
        Decimation factor, defaults to 0.
        If you use large number, the calculation will be accelerated.
    confidence : `float`, optional
        Confidence interval to determine the threshold for the weak texture.
        In this algorithm, this value is usually set the value very close to one.
        Defaults to 0.99.
    iterations : `int`, optional
        Number of iterations,  defaults to 3.

    Returns
    -------
    `tuple`
        A tuple containing the estimated noise levels,  threshold to extract weak texture
        patches at the last iteration, and number of extracted weak texture patches.
    """
    if len(img.shape) < 3:
        img = np.expand_dims(img, 2)

    nlevel = np.ndarray(img.shape[2])
    thresh = np.ndarray(img.shape[2])
    num = np.ndarray(img.shape[2])

    kh = np.expand_dims(np.expand_dims(np.array([-0.5, 0, 0.5]), 0), 2)
    imgh = correlate(img, kh, mode="nearest")
    imgh = imgh[:, 1 : imgh.shape[1] - 1, :]
    imgh = imgh * imgh

    kv = np.expand_dims(np.vstack(np.array([-0.5, 0, 0.5])), 2)
    imgv = correlate(img, kv, mode="nearest")
    imgv = imgv[1 : imgv.shape[0] - 1, :, :]
    imgv = imgv * imgv

    Dh = conv2d_matrix(np.squeeze(kh, 2), patchsize, patchsize)
    Dv = conv2d_matrix(np.squeeze(kv, 2), patchsize, patchsize)

    DD = np.transpose(Dh) @ Dh + np.transpose(Dv) @ Dv

    r = np.double(np.linalg.matrix_rank(DD))
    Dtr = np.trace(DD)

    tau0 = gamma.ppf(confidence, r / 2, scale=(2 * Dtr / r))

    for cha in range(img.shape[2]):
        X = view_as_windows(img[:, :, cha], (patchsize, patchsize))
        X = X.reshape(int(X.size / patchsize**2), patchsize**2, order="F").transpose()

        Xh = view_as_windows(imgh[:, :, cha], (patchsize, patchsize - 2))
        Xh = Xh.reshape(
            int(Xh.size / ((patchsize - 2) * patchsize)), ((patchsize - 2) * patchsize), order="F"
        ).transpose()

        Xv = view_as_windows(imgv[:, :, cha], (patchsize - 2, patchsize))
        Xv = Xv.reshape(
            int(Xv.size / ((patchsize - 2) * patchsize)), ((patchsize - 2) * patchsize), order="F"
        ).transpose()

        Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=0), axis=0), 0)

        if decim > 0:
            XtrX = np.transpose(np.concatenate((Xtr, X), axis=0))
            XtrX = np.transpose(
                XtrX[
                    XtrX[:, 0].argsort(),
                ]
            )
            p = np.floor(XtrX.shape[1] / (decim + 1))
            p = np.expand_dims(np.arange(0, p) * (decim + 1), 0)
            Xtr = XtrX[0, p.astype("int")]
            X = np.squeeze(XtrX[1 : XtrX.shape[1], p.astype("int")])

        # noise level estimation
        tau = np.inf

        if X.shape[1] < X.shape[0]:
            sig2 = 0
        else:
            cov = (X @ np.transpose(X)) / (X.shape[1] - 1)
            d = np.flip(np.linalg.eig(cov)[0], axis=0)
            sig2 = d[0]

        for _ in range(1, iterations):
            # weak texture selection
            tau = sig2 * tau0
            p = Xtr < tau
            Xtr = Xtr[p]
            X = X[:, np.squeeze(p)]

            # noise level estimation
            if X.shape[1] < X.shape[0]:
                break

            cov = (X @ np.transpose(X)) / (X.shape[1] - 1)
            d = np.flip(np.linalg.eig(cov)[0], axis=0)
            sig2 = d[0]

        nlevel[cha] = np.sqrt(sig2)
        thresh[cha] = tau
        num[cha] = X.shape[1]

    # clean up
    img = np.squeeze(img)

    return nlevel, thresh, num


def conv2d_matrix(H, rows, columns):
    """
    Specialized 2D convolution matrix generation.

    Parameters
    ----------
    H : `numpy.ndarray`
        Input matrix.
    rows : `numpy.ndarray`
        Rows in convolution matrix.
    columns : `numpy.ndarray`
        Columns in convolution matrix.

    Returns
    -------
    T : `numpy.ndarray`
        The new convoluted matrix.
    """
    s = np.shape(H)
    rows = int(rows)
    columns = int(columns)

    matr_row = rows - s[0] + 1
    matr_column = columns - s[1] + 1

    T = np.zeros([matr_row * matr_column, rows * columns])

    k = 0
    for i in range(matr_row):
        for j in range(matr_column):
            for p in range(s[0]):
                start = (i + p) * columns + j
                T[k, start : start + s[1]] = H[p, :]

            k += 1
    return T


def weak_texture_mask(img, patchsize, thresh):
    """
    Calculates the weak texture mask.

    Parameters
    ----------
    img: `numpy.ndarray`
        Single Numpy image array.
    patchsize : `int`, optional
        Patch size, defaults to 7.
    thresh: `numpy.ndarray`
        Threshold to extract weak texture patches at the last iteration.

    Returns
    -------
    mask: `numpy.ndarray`
        Weak-texture mask. 0 and 1 represent non-weak-texture and weak-texture regions, respectively.
    """
    if img.ndim < 3:
        img = np.expand_dims(img, 2)

    kh = np.expand_dims(np.transpose(np.vstack(np.array([-0.5, 0, 0.5]))), 2)
    imgh = correlate(img, kh, mode="nearest")
    imgh = imgh[:, 1 : imgh.shape[1] - 1, :]
    imgh = imgh * imgh

    kv = np.expand_dims(np.vstack(np.array([-0.5, 0, 0.5])), 1)
    imgv = correlate(img, kv, mode="nearest")
    imgv = imgv[1 : imgv.shape[0] - 1, :, :]
    imgv = imgv * imgv

    s = img.shape
    msk = np.zeros_like(img)

    for cha in range(s[2]):
        m = view_as_windows(img[:, :, cha], (patchsize, patchsize))
        m = np.zeros_like(m.reshape(int(m.size / patchsize**2), patchsize**2, order="F").transpose())

        Xh = view_as_windows(imgh[:, :, cha], (patchsize, patchsize - 2))
        Xh = Xh.reshape(
            int(Xh.size / ((patchsize - 2) * patchsize)), ((patchsize - 2) * patchsize), order="F"
        ).transpose()

        Xv = view_as_windows(imgv[:, :, cha], (patchsize - 2, patchsize))
        Xv = Xv.reshape(
            int(Xv.size / ((patchsize - 2) * patchsize)), ((patchsize - 2) * patchsize), order="F"
        ).transpose()

        Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=0), axis=0), 0)

        p = Xtr < thresh[cha]
        ind = 0

        for col in range(0, s[1] - patchsize + 1):
            for row in range(0, s[0] - patchsize + 1):
                if p[:, ind]:
                    msk[row : row + patchsize - 1, col : col + patchsize - 1, cha] = 1
                ind = ind + 1

    # clean up
    img = np.squeeze(img)

    return np.squeeze(msk)
