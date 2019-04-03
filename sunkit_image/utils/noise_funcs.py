'''
This module implements a series of functions for noise level estimation
'''

import numpy as np
from scipy.stats import gamma
from skimage.util import view_as_windows
from scipy.ndimage import correlate

__all__ = ["NoiseLevelEstimation"]

#TODO: Variable type checks on function calls
#TODO: Testing
#TODO: Documentation and comments

def noise_estimation(img, patchsize=7, decim=0, conf=1 - 1e-6, itr=3):
   
    nlevel, th, num = noiselevel(img, patchsize, decim, conf, itr)
    mask = weaktexturemask(img, patchsize, th)

def noiselevel(img, patchsize, decim, conf, itr):
    """
    Calculates the noise level of the input array.
    """
    if len(img.shape) < 3:
        img = np.expand_dims(img, 2)

    nlevel = np.ndarray(img.shape[2])
    th = np.ndarray(img.shape[2])
    num = np.ndarray(img.shape[2])

    kh = np.expand_dims(np.expand_dims(np.array([-0.5, 0, 0.5]), 0), 2)
    imgh = correlate(img, kh, mode="nearest")
    imgh = imgh[:, 1 : imgh.shape[1] - 1, :]
    imgh = imgh * imgh

    kv = np.expand_dims(np.vstack(np.array([-0.5, 0, 0.5])), 2)
    imgv = correlate(img, kv, mode="nearest")
    imgv = imgv[1 : imgv.shape[0] - 1, :, :]
    imgv = imgv * imgv

    Dh = np.matrix(convmtx2(np.squeeze(kh, 2), patchsize, patchsize))
    Dv = np.matrix(convmtx2(np.squeeze(kv, 2), patchsize, patchsize))

    DD = Dh.getH() * Dh + Dv.getH() * Dv

    r = np.double(np.linalg.matrix_rank(DD))
    Dtr = np.trace(DD)

    tau0 = gamma.ppf(conf, r / 2, scale=(2 * Dtr / r))

    for cha in range(img.shape[2]):
        X = view_as_windows(img[:, :, cha], (patchsize, patchsize))
        X = X.reshape(
            np.int(X.size / patchsize ** 2), patchsize ** 2, order="F"
        ).transpose()

        Xh = view_as_windows(imgh[:, :, cha], (patchsize, patchsize - 2))
        Xh = Xh.reshape(
            np.int(Xh.size / ((patchsize - 2) * patchsize)),
            ((patchsize - 2) * patchsize),
            order="F",
        ).transpose()

        Xv = view_as_windows(imgv[:, :, cha], (patchsize - 2, patchsize))
        Xv = Xv.reshape(
            np.int(Xv.size / ((patchsize - 2) * patchsize)),
            ((patchsize - 2) * patchsize),
            order="F",
        ).transpose()

        Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=0), axis=0), 0)
        
        if decim > 0:
            XtrX = np.transpose(np.concatenate((Xtr, X), axis=0))
            XtrX = np.transpose(XtrX[XtrX[:, 0].argsort(),])
            p = np.floor(XtrX.shape[1] / (decim + 1))
            p = np.expand_dims(np.arange(0, p) * (decim + 1), 0)
            Xtr = XtrX[0, p.astype("int")]
            X = np.squeeze(XtrX[1 : XtrX.shape[1], p.astype("int")])

        # noise level estimation
        tau = np.inf

        if X.shape[1] < X.shape[0]:
            sig2 = 0
        else:
            cov = (np.asmatrix(X) @ np.asmatrix(X).getH()) / (X.shape[1] - 1)
            d = np.flip(np.linalg.eig(cov)[0], axis=0)
            sig2 = d[0]

        for i in range(1, itr):
            # weak texture selection
            tau = sig2 * tau0
            p = Xtr < tau
            Xtr = Xtr[p]
            X = X[:, np.squeeze(p)]

            # noise level estimation
            if X.shape[1] < X.shape[0]:
                break

            cov = (np.asmatrix(X) @ np.asmatrix(X).getH()) / (X.shape[1] - 1)
            d = np.flip(np.linalg.eig(cov)[0], axis=0)
            sig2 = d[0]

        nlevel[cha] = np.sqrt(sig2)
        th[cha] = tau
        num[cha] = X.shape[1]

    # clean up
    img = np.squeeze(img)

    return nlevel, th, num

def convmtx2(H, m, n):
    """
    Specialized 2D convolution matrix generation.

    Parameters
    ----------
    H : `numpy.ndarray`
        Input matrix.
    m : `numpy.ndarray`
        Rows in convolution matrix.
    n : `numpy.ndarray`
        Columns in convolution matrix.

    Returns
    -------
    T : `numpy.ndarray`
        The new convoluted matrix.
    """
    s = np.shape(H)
    T = np.zeros([(m - s[0] + 1) * (n - s[1] + 1), m * n])

    k = 0
    for i in range((m - s[0] + 1)):
        for j in range((n - s[1] + 1)):
            for p in range(s[0]):
                T[k, (i + p) * n + j : (i + p) * n + j + 1 + s[1] - 1] = H[p, :]
                k = k + 1

    return T


def weaktexturemask(img, patchsize, th):
    """
    Calculates the weak texture mask.
    """
    if len(img.shape) < 3:
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
        m = np.zeros_like(
            m.reshape(
                np.int(m.size / patchsize ** 2), patchsize ** 2, order="F"
            ).transpose()
        )

        Xh = view_as_windows(imgh[:, :, cha], (patchsize, patchsize - 2))
        Xh = Xh.reshape(
            np.int(Xh.size / ((patchsize - 2) * patchsize)),
            ((patchsize - 2) * patchsize),
            order="F",
        ).transpose()

        Xv = view_as_windows(imgv[:, :, cha], (patchsize - 2, patchsize))
        Xv = Xv.reshape(
            np.int(Xv.size / ((patchsize - 2) * patchsize)),
            ((patchsize - 2) * patchsize),
            order="F",
        ).transpose()

        Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=0), axis=0), 0)

        p = Xtr < th[cha]
        ind = 0

        for col in range(0, s[1] - patchsize + 1):
            for row in range(0, s[0] - patchsize + 1):
                if p[:, ind]:
                    msk[
                        row : row + patchsize - 1,
                        col : col + patchsize - 1,
                        cha,
                    ] = 1
                ind = ind + 1

    # clean up
    img = np.squeeze(img)

    return np.squeeze(msk)
