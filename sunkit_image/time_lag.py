"""
This module contains functions for calculating the cross-correlation and time
lag between intensity cubes.
"""
import numpy as np

import astropy.units as u

__all__ = [
    "cross_correlation",
    "get_lags",
    "max_cross_correlation",
    "time_lag",
]


@u.quantity_input
def get_lags(time: u.s):
    """
    Convert an array of evenly spaced times to an array of time lags
    evenly spaced between ``-max(time)`` and ``max(time)``.
    """
    delta_t = np.diff(time)
    if not np.allclose(delta_t, delta_t[0]):
        raise ValueError("Times must be evenly sampled")
    delta_t = delta_t.cumsum(axis=0)
    return np.hstack([-delta_t[::-1], np.array([0]), delta_t])


@u.quantity_input
def cross_correlation(signal_a, signal_b, lags: u.s):
    r"""
    Compute cross-correlation between two signals, as a function of lag

    By the convolution theorem the cross-correlation between two signals
    can be computed as,

    .. math::

        \mathcal{C}_{AB}(\tau) &= \mathcal{I}_A(t)\star\mathcal{I}_B(t) \\
        &= \mathcal{I}_A(-t)\ast\mathcal{I}_B(t) \\
        &= \mathscr{F}^{-1}\{\mathscr{F}\{\mathcal{I}_A(-t)\}\mathscr{F}\{\mathcal{I}_B(t)\}\}

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
        and must have length ``(len(lags) + 1)/2``.
    signal_b : array-like
        Must have the same dimensions as ``signal_a``.
    lags : `~astropy.units.Quantity`
        Evenly spaced time lags corresponding to the time dimension of
        ``signal_a`` and ``signal_b`` running from ``-max(time)`` to
        ``max(time)``. This is easily constructed using :func:`get_lags`

    Returns
    -------
    array-like
        Cross-correlation as a function of ``lags``. The first dimension will be
        the same as that of ``lags`` and the subsequent dimensions will be
        consistent with dimensions of ``signal_a`` and ``signal_b``.

    See Also
    ---------
    get_lags
    time_lag
    max_cross_correlation

    References
    -----------
    * https://en.wikipedia.org/wiki/Convolution_theorem
    * Viall, N.M. and Klimchuk, J.A.
      Evidence for Widespread Cooling in an Active Region Observed with the SDO Atmospheric Imaging Assembly
      ApJ, 753, 35, 2012
      (https://doi.org/10.1088/0004-637X/753/1/35)
    * Appendix C in Barnes, W.T., Bradshaw, S.J., Viall, N.M.
      Understanding Heating in Active Region Cores through Machine Learning. I. Numerical Modeling and Predicted Observables
      ApJ, 880, 56, 2019
      (https://doi.org/10.3847/1538-4357/ab290c)
    """
    # NOTE: it is assumed that the arrays have already been appropriately
    # interpolated and chunked (if using Dask)
    delta_lags = np.diff(lags)
    if not u.allclose(delta_lags, delta_lags[0]):
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


def _dask_check(signal, lags):
    # This is to avoid having to specify time as a Dask array so that it
    # can be specified as a quantity and so that the time lag can be
    # returned as a Dask quantity and not computed eagerly
    try:
        import dask.array  # do this here so that Dask is not a hard requirement
    except ImportError:
        return lags
    if isinstance(signal, dask.array.Array):
        return dask.array.from_array(lags, chunks=lags.shape)
    else:
        return lags


@u.quantity_input
def time_lag(signal_a, signal_b, time: u.s, lag_bounds: (u.s, None) = None, **kwargs):
    r"""
    Compute the time lag that maximizes the cross-correlation
    between ``signal_a`` and ``signal_b``.

    For a pair of signals :math:`A,B`, e.g. time series from two EUV channels
    on AIA, the time lag is the lag which maximizes the cross-correlation,

    .. math::

        \tau_{AB} = \mathop{\mathrm{arg\,max}}_{\tau}\mathcal{C}_{AB},

    where :math:`\mathcal{C}_{AB}` is the cross-correlation as a function of
    lag (computed in :func:`cross_correlation`). Qualitatively, this can be
    thought of as how much `signal_a` needs to be shifted in time to best
    "match" `signal_b`. Note that the sign of :math:`\\tau_{AB}`` is determined
    by the ordering of the two signals such that,

    .. math::

        \tau_{AB} = -\tau_{BA}.

    Parameters
    ----------
    signal_a : array-like
        The first dimension must be the same length as ``time``.
    signal_b : array-like
        Must have the same dimensions as ``signal_a``.
    time : `~astropy.units.Quantity`
        Time array corresponding to the intensity time series
        ``signal_a`` and ``signal_b``.
    lag_bounds : `~astropy.units.Quantity`, optional
        Minimum and maximum lag to consider when finding the time
        lag that maximizes the cross-correlation. This is useful
        for minimizing boundary effects.

    Other Parameters
    ----------------
    pre_check_hook : function
        Function to apply to `lags` array prior to selecting maximum lags. This
        is usful when `signal_a` and `signal_b` are of a type besides `~numpy.ndarray`.
        This function should accept `signal_a` and `lags` and return an array that
        looks like `lags`.
    post_check_hook : function
        Function to apply to the resulting time lag result. This should take in the
        result of the time lag selection and return something that an array that looks
        like the time lag selection.

    Returns
    -------
    array-like
        Lag which maximizes the cross-correlation. The dimensions will be
        consistent with those of ``signal_a`` and ``signal_b``, i.e. if the
        input arrays are of dimension ``(K,M,N)``, the resulting array
        will have dimensions ``(M,N)``. Similarly, if the input signals
        are one-dimensional time series ``(K,)``, the result will have
        dimension ``(1,)``.

    References
    ----------
    * Viall, N.M. and Klimchuk, J.A.
      Evidence for Widespread Cooling in an Active Region Observed with the SDO Atmospheric Imaging Assembly
      ApJ, 753, 35, 2012
      (https://doi.org/10.1088/0004-637X/753/1/35)
    """
    pre_check = kwargs.get("pre_check_hook", _dask_check)
    post_check = kwargs.get("post_check_hook", lambda x: x)
    lags = get_lags(time)
    cc = cross_correlation(signal_a, signal_b, lags)
    start, stop = _get_bounds_indices(lags, lag_bounds)
    i_max_cc = cc[start:stop].argmax(axis=0)
    # The flatten + reshape is needed here because Dask does not like
    # "fancy" multidimensional indexing
    lags = pre_check(signal_a, lags)
    return post_check(lags[start:stop][i_max_cc.flatten()].reshape(i_max_cc.shape))


@u.quantity_input
def max_cross_correlation(signal_a, signal_b, time: u.s, lag_bounds: (u.s, None) = None):
    """
    Compute the maximum value of the cross-correlation between ``signal_a``
    and ``signal_b``.

    This is the maximum value of the cross-correlation as a function of
    lag (computed in :func:`cross_correlation`). This will always be between
    -1 (perfectly anti-correlated) and +1 (perfectly correlated) though
    in practice is nearly always between 0 and +1.

    Parameters
    ----------
    signal_a : array-like
        The first dimension must be the same length as ``time``.
    signal_b : array-like
        Must have the same dimensions as ``signal_a``.
    time : array-like
        Time array corresponding to the intensity time series
        ``signal_a`` and ``signal_b``.
    lag_bounds : `tuple`, optional
        Minimum and maximum lag to consider when finding the time
        lag that maximizes the cross-correlation. This is useful
        for minimizing boundary effects.

    Returns
    -------
    array-like
        Maximum value of the cross-correlation. The dimensions will be
        consistent with those of ``signal_a`` and ``signal_b``, i.e. if the
        input arrays are of dimension ``(K,M,N)``, the resulting array
        will have dimensions ``(M,N)``. Similarly, if the input signals
        are one-dimensional time series ``(K,)``, the result will have
        dimension ``(1,)``.

    References
    ----------
    * Viall, N.M. and Klimchuk, J.A.
      Evidence for Widespread Cooling in an Active Region Observed with the SDO Atmospheric Imaging Assembly
      ApJ, 753, 35, 2012
      (https://doi.org/10.1088/0004-637X/753/1/35)
    """
    lags = get_lags(time)
    cc = cross_correlation(signal_a, signal_b, lags)
    start, stop = _get_bounds_indices(lags, lag_bounds)
    return cc[start:stop].max(axis=0)
