"""
This module implements a class for noise level estimation.
"""

import numpy as np
from scipy.ndimage import correlate
from scipy.stats import gamma
from skimage.util import view_as_windows

__all__ = ["NoiseLevelEstimation"]


class NoiseLevelEstimation:
    def __init__(self, img, patchsize=7, decim=0, conf=1 - 1e-6, itr=3):
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
        conf : `float`, optional
            Confidence interval to determine the threshold for the weak texture.
            In this algorithm, this value is usually set the value very close to one.
            Defaults to 0.99.
        itr : `int`, optional
            Number of iterations,  defaults to 3.

        Attributes
        ----------
        nlevel: `numpy.ndarray`
            Estimated noise levels.
        th: `numpy.ndarray`
            Threshold to extract weak texture patches at the last iteration.
        num: `numpy.ndarray`
            Number of extracted weak texture patches at the last iteration.
        mask: `numpy.ndarray`
            Weak-texture mask.
            0 and 1 represent non-weak-texture and weak-texture regions, respectively.

        Examples
        --------
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> noisy_image_array = np.random.randn(100, 100)
        >>> estimate = NoiseLevelEstimation(noisy_image_array, patchsize=11, itr=10)
        >>> estimate.mask
        array([[1., 1., 1., ..., 1., 1., 0.],
            [1., 1., 1., ..., 1., 1., 0.],
            [1., 1., 1., ..., 1., 1., 0.],
            ...,
            [1., 1., 1., ..., 1., 1., 0.],
            [1., 1., 1., ..., 1., 1., 0.],
            [0., 0., 0., ..., 0., 0., 0.]])
        >>> estimate.nlevel
        array([1.0014616])
        >>> estimate.th
        array([173.61530607])
        >>> estimate.num
         array([8100.])

        References
        ----------
        * Xinhao Liu, Masayuki Tanaka and Masatoshi Okutomi
          Noise Level Estimation Using Weak Textured Patches of a Single Noisy Image
          IEEE International Conference on Image Processing (ICIP), 2012.

        * Xinhao Liu, Masayuki Tanaka and Masatoshi Okutomi
          Single-Image Noise Level Estimation for Blind Denoising Noisy Image
          IEEE Transactions on Image Processing, Vol.22, No.12, pp.5226-5237, December, 2013.
        """

        self.img = img
        self.patchsize = patchsize
        self.decim = decim
        self.conf = conf
        self.itr = itr

        self.nlevel, self.th, self.num = self.noiselevel()
        self.mask = self.weaktexturemask()

    def noiselevel(self):
        """
        Calculates the noise level of the input array.
        """
        if len(self.img.shape) < 3:
            self.img = np.expand_dims(self.img, 2)

        nlevel = np.ndarray(self.img.shape[2])
        th = np.ndarray(self.img.shape[2])
        num = np.ndarray(self.img.shape[2])

        kh = np.expand_dims(np.expand_dims(np.array([-0.5, 0, 0.5]), 0), 2)
        imgh = correlate(self.img, kh, mode="nearest")
        imgh = imgh[:, 1 : imgh.shape[1] - 1, :]
        imgh = imgh * imgh

        kv = np.expand_dims(np.vstack(np.array([-0.5, 0, 0.5])), 2)
        imgv = correlate(self.img, kv, mode="nearest")
        imgv = imgv[1 : imgv.shape[0] - 1, :, :]
        imgv = imgv * imgv

        Dh = np.matrix(self.convmtx2(np.squeeze(kh, 2), self.patchsize, self.patchsize))
        Dv = np.matrix(self.convmtx2(np.squeeze(kv, 2), self.patchsize, self.patchsize))

        DD = Dh.getH() * Dh + Dv.getH() * Dv

        r = np.double(np.linalg.matrix_rank(DD))
        Dtr = np.trace(DD)

        tau0 = gamma.ppf(self.conf, r / 2, scale=(2 * Dtr / r))

        for cha in range(self.img.shape[2]):
            X = view_as_windows(self.img[:, :, cha], (self.patchsize, self.patchsize))
            X = X.reshape(
                np.int(X.size / self.patchsize ** 2), self.patchsize ** 2, order="F"
            ).transpose()

            Xh = view_as_windows(imgh[:, :, cha], (self.patchsize, self.patchsize - 2))
            Xh = Xh.reshape(
                np.int(Xh.size / ((self.patchsize - 2) * self.patchsize)),
                ((self.patchsize - 2) * self.patchsize),
                order="F",
            ).transpose()

            Xv = view_as_windows(imgv[:, :, cha], (self.patchsize - 2, self.patchsize))
            Xv = Xv.reshape(
                np.int(Xv.size / ((self.patchsize - 2) * self.patchsize)),
                ((self.patchsize - 2) * self.patchsize),
                order="F",
            ).transpose()

            Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=0), axis=0), 0)

            if self.decim > 0:
                XtrX = np.transpose(np.concatenate((Xtr, X), axis=0))
                XtrX = np.transpose(XtrX[XtrX[:, 0].argsort(),])
                p = np.floor(XtrX.shape[1] / (self.decim + 1))
                p = np.expand_dims(np.arange(0, p) * (self.decim + 1), 0)
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

            for i in range(1, self.itr):
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
        self.img = np.squeeze(self.img)

        return nlevel, th, num

    def convmtx2(self, H, m, n):
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

    def weaktexturemask(self):
        """
        Calculates the weak texture mask.
        """
        if len(self.img.shape) < 3:
            self.img = np.expand_dims(self.img, 2)

        kh = np.expand_dims(np.transpose(np.vstack(np.array([-0.5, 0, 0.5]))), 2)
        imgh = correlate(self.img, kh, mode="nearest")
        imgh = imgh[:, 1 : imgh.shape[1] - 1, :]
        imgh = imgh * imgh

        kv = np.expand_dims(np.vstack(np.array([-0.5, 0, 0.5])), 1)
        imgv = correlate(self.img, kv, mode="nearest")
        imgv = imgv[1 : imgv.shape[0] - 1, :, :]
        imgv = imgv * imgv

        s = self.img.shape
        msk = np.zeros_like(self.img)

        for cha in range(s[2]):
            m = view_as_windows(self.img[:, :, cha], (self.patchsize, self.patchsize))
            m = np.zeros_like(
                m.reshape(
                    np.int(m.size / self.patchsize ** 2), self.patchsize ** 2, order="F"
                ).transpose()
            )

            Xh = view_as_windows(imgh[:, :, cha], (self.patchsize, self.patchsize - 2))
            Xh = Xh.reshape(
                np.int(Xh.size / ((self.patchsize - 2) * self.patchsize)),
                ((self.patchsize - 2) * self.patchsize),
                order="F",
            ).transpose()

            Xv = view_as_windows(imgv[:, :, cha], (self.patchsize - 2, self.patchsize))
            Xv = Xv.reshape(
                np.int(Xv.size / ((self.patchsize - 2) * self.patchsize)),
                ((self.patchsize - 2) * self.patchsize),
                order="F",
            ).transpose()

            Xtr = np.expand_dims(np.sum(np.concatenate((Xh, Xv), axis=0), axis=0), 0)

            p = Xtr < self.th[cha]
            ind = 0

            for col in range(0, s[1] - self.patchsize + 1):
                for row in range(0, s[0] - self.patchsize + 1):
                    if p[:, ind]:
                        msk[row : row + self.patchsize - 1, col : col + self.patchsize - 1, cha] = 1
                    ind = ind + 1

        # clean up
        self.img = np.squeeze(self.img)

        return np.squeeze(msk)
