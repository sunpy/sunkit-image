"""
This module contains functions for calculating the cross-correlation and time
lag between intensity cubes.
"""
import numpy as np

__all__ = [
    "cross_correlation",
    "get_lags",
    "max_cross_correlation",
    "time_lag",
]


def get_lags(time):
    """"""
    delta_t = np.diff(time)
    if not np.allclose(delta_t, delta_t[0]):
        raise ValueError("Times must be evenly sampled")
    delta_t = delta_t.cumsum(axis=0)
    return np.hstack([-delta_t[::-1], np.array([0]), delta_t])


def cross_correlation(signal_a, signal_b, lags):
    """
    Compute cross-correlation between two signals, as a function of lag

    By the convolution theorem the cross-correlation between two signals
    can be computed as,

    .. math::

        \mathcal{C}_{AB}(\\tau) &= \mathcal{I}_A(t)\star\mathcal{I}_B(t) = \mathcal{I}_A(-t)\\ast\mathcal{I}_B(t) \\
        &= \inversefourier{\\fourier{\mathcal{I}_A(-t)}\\fourier{\mathcal{I}_B(t)}}

    where each signal has been centered and scaled by its mean and standard
    deviation,

    .. math::

        \mathcal{I}_c(t)=\frac{I_c(t)-\bar{I}_c}{\sigma_{c}}

    Additionally, :math:`\mathcal{C}_{AB}` is normalized by the length of
    the time series.

    Parameters
    -----------
    signal_a : array-like
        The first dimension should correspond to the time dimension
        and must have length ``(len(lags) + 1)/2``
    signal_b : array-like
        Must have the same dimensions as `signal_a`
    lags : array-like
        Evenly spaced time lags corresponding to the time dimension of
        `signal_a` and `signal_b` running from ``-max(time)`` to
        ``max(time)``. This is easily constructed using :func:`get_lags`

    See Also
    ---------
    get_lags
    time_lag
    max_cross_correlation

    References
    -----------
    - https://en.wikipedia.org/wiki/Convolution_theorem
    - Viall, N.M. and Klimchuk, J.A.
      Evidence for Widespread Cooling in an Active Region Observed with the SDO Atmospheric Imaging Assembly
      ApJ, 753, 35, 2012
      (https://doi.org/10.1088/0004-637X/753/1/35)
    - Appendix C in Barnes, W.T., Bradshaw, S.J., Viall, N.M.
      Understanding Heating in Active Region Cores through Machine Learning. I. Numerical Modeling and Predicted Observables
      ApJ, 880, 56, 2019
      (https://doi.org/10.3847/1538-4357/ab290c)
    """
    # NOTE: it is assumed that the arrays have already been appropriately
    # interpolated and chunked (if using Dask)
    delta_lags = np.diff(lags)
    if not np.allclose(delta_lags, delta_lags[0]):
        raise ValueError("Lags must be evenly sampled")
    n_time = (lags.shape[0] + 1) // 2
    if signal_a.shape != signal_b.shape:
        raise ValueError("Signals must have same shape.")
    if signal_a.shape[0] != n_time:
        raise ValueError("First dimension of signal must be equal in length to time array.")
    # Reverse the first timeseries
    signal_a = signal_a[::-1]
    # Normalize by mean and standard deviation
    std_a = signal_a.std(axis=0)
    std_a = np.where(std_a == 0, 1, std_a)  # avoid dividing by zero
    v_a = (signal_a - signal_a.mean(axis=0)[np.newaxis]) / std_a[np.newaxis]
    std_b = signal_b.std(axis=0)
    std_b = np.where(std_b == 0, 1, std_b)
    v_b = (signal_b - signal_b.mean(axis=0)[np.newaxis]) / std_b[np.newaxis]
    # Cross-correlation is inverse of product of FFTS (by convolution theorem)
    fft_a = np.fft.rfft(v_a, axis=0, n=lags.shape[0])
    fft_b = np.fft.rfft(v_b, axis=0, n=lags.shape[0])
    cc = np.fft.irfft(fft_a * fft_b, axis=0, n=lags.shape[0])
    # Normalize by the length of the timeseries
    return cc / signal_a.shape[0]


def _get_bounds_indices(lags, bounds):
    # The start and stop indices are computed in this way
    # because Dask does not like "fancy" multidimensional indexing
    if bounds is not None:
        (indices,) = np.where(np.logical_and(lags >= bounds[0], lags <= bounds[1]))
        start = indices[0]
        stop = indices[-1] + 1
    else:
        start = 0
        stop = lags.shape[0] + 1
    return start, stop


def time_lag(signal_a, signal_b, time, lag_bounds=None):
    """"""
    lags = get_lags(time)
    cc = cross_correlation(signal_a, signal_b, lags)
    start, stop = _get_bounds_indices(lags, lag_bounds)
    i_max_cc = cc[start:stop].argmax(axis=0)
    # The flatten + reshape is needed here because Dask does not like
    # "fancy" multidimensional indexing
    return lags[start:stop][i_max_cc.flatten()].reshape(i_max_cc.shape)


def max_cross_correlation(signal_a, signal_b, time, lag_bounds=None):
    """"""
    lags = get_lags(time)
    cc = cross_correlation(signal_a, signal_b, lags)
    start, stop = _get_bounds_indices(lags, lag_bounds)
    return cc[start:stop].max(axis=0)
