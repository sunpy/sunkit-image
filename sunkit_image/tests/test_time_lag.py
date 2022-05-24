import dask.array
import numpy as np
import pytest

import astropy.units as u

from sunkit_image.time_lag import cross_correlation, get_lags, max_cross_correlation, time_lag


@pytest.mark.parametrize("shape_in,shape_out", [((20, 5, 5), (39, 5, 5)), ((100, 10), (199, 10)), ((1000,), (1999,))])
def test_cross_correlation_array_shapes(shape_in, shape_out):
    s_a = np.random.rand(*shape_in)
    s_b = np.random.rand(*shape_in)
    time = np.linspace(0, 1, shape_in[0]) * u.s
    lags = get_lags(time)
    cc = cross_correlation(s_a, s_b, lags)
    assert cc.shape == shape_out


@pytest.mark.parametrize("shape", [((5, 5)), ((10,)), ((1,))])
def test_max_cc_time_lag_array_shapes(shape):
    time = np.linspace(0, 1, 10) * u.s
    shape_in = time.shape + shape
    s_a = np.random.rand(*shape_in)
    s_b = np.random.rand(*shape_in)
    max_cc = max_cross_correlation(s_a, s_b, time)
    tl = time_lag(s_a, s_b, time)
    assert max_cc.shape == shape
    assert tl.shape == shape


@pytest.mark.parametrize("shape", [((5, 5)), ((10,)), ((1,))])
def test_time_lag_calculation(shape):
    def gaussian_pulse(x, x0, sigma):
        return np.exp(-((x - x0) ** 2) / (2 * sigma**2))

    time = np.linspace(0, 1, 500) * u.s
    s_a = gaussian_pulse(time, 0.4 * u.s, 0.02 * u.s)
    s_b = gaussian_pulse(time, 0.6 * u.s, 0.02 * u.s)
    s_a = s_a * np.ones(shape + time.shape)
    s_b = s_b * np.ones(shape + time.shape)
    tl = time_lag(s_a.T, s_b.T, time)
    assert u.allclose(tl, 0.2 * u.s, rtol=5e-3)


@pytest.mark.parametrize(
    "shape_in",
    [
        ((20, 5, 5)),
        ((100, 10)),
        ((1000, 1)),
    ],
)
def test_preserve_array_types(shape_in):
    s_a = np.random.rand(*shape_in)
    s_b = np.random.rand(*shape_in)
    time = np.linspace(0, 1, shape_in[0]) * u.s
    # Numpy arrays
    max_cc = max_cross_correlation(s_a, s_b, time)
    tl = time_lag(s_a, s_b, time)
    assert isinstance(max_cc, np.ndarray)
    assert isinstance(tl, u.Quantity)
    # Dask arrays
    s_a = dask.array.from_array(s_a)
    s_b = dask.array.from_array(s_b)
    max_cc = max_cross_correlation(s_a, s_b, time)
    tl = time_lag(s_a, s_b, time)
    assert isinstance(max_cc, dask.array.Array)
    assert isinstance(tl, dask.array.Array)


@pytest.mark.parametrize(
    "shape_in",
    [
        ((20, 5, 5)),
        ((100, 10)),
        ((1000, 1)),
    ],
)
def test_dask_numpy_consistent(shape_in):
    s_a = np.random.rand(*shape_in)
    s_b = np.random.rand(*shape_in)
    time = np.linspace(0, 1, shape_in[0]) * u.s
    max_cc = max_cross_correlation(s_a, s_b, time)
    tl = time_lag(s_a, s_b, time)
    s_a = dask.array.from_array(s_a)
    s_b = dask.array.from_array(s_b)
    max_cc_dask = max_cross_correlation(s_a, s_b, time)
    tl_dask = time_lag(s_a, s_b, time)
    assert u.allclose(tl, tl_dask.compute(), rtol=0.0, atol=None)
    assert u.allclose(max_cc, max_cc_dask.compute(), rtol=0.0, atol=None)


@pytest.mark.parametrize(
    "shape_in",
    [
        ((20, 5, 5)),
        ((100, 10)),
        ((1000, 1)),
    ],
)
def test_quantity_numpy_consistent(shape_in):
    # Test that Quantities can be used as inputs for the signals and that
    # it gives equivalent results to using bare numpy arrays
    s_a = np.random.rand(*shape_in) * u.ct / u.s
    s_b = np.random.rand(*shape_in) * u.ct / u.s
    time = np.linspace(0, 1, shape_in[0]) * u.s
    for func in [time_lag, max_cross_correlation]:
        result_numpy = func(s_a.value, s_b.value, time)
        result_quantity = func(s_a, s_b, time)
        assert u.allclose(result_numpy, result_quantity, rtol=0.0, atol=None)


@pytest.mark.parametrize(
    "shape_a,shape_b,lags,exception",
    [
        ((10, 1), (10, 1), np.array([-1, -0.5, 0.1, 1]) * u.s, "Lags must be evenly sampled"),
        ((10, 2, 3), (10, 2, 4), np.linspace(-1, 1, 19) * u.s, "Signals must have same shape."),
        (
            (20, 5),
            (20, 5),
            np.linspace(-1, 1, 10) * u.s,
            "First dimension of signal must be equal in length to time array.",
        ),
    ],
)
def test_exceptions(shape_a, shape_b, lags, exception):
    s_a = np.random.rand(*shape_a)
    s_b = np.random.rand(*shape_b)
    with pytest.raises(ValueError, match=exception):
        _ = cross_correlation(s_a, s_b, lags)


def test_bounds():
    time = np.linspace(0, 1, 10) * u.s
    shape = time.shape + (5, 5)
    s_a = np.random.rand(*shape)
    s_b = np.random.rand(*shape)
    bounds = (-0.5, 0.5) * u.s
    max_cc = max_cross_correlation(s_a, s_b, time, lag_bounds=bounds)
    tl = time_lag(s_a, s_b, time, lag_bounds=bounds)
    assert isinstance(max_cc, np.ndarray)
    assert isinstance(tl, u.Quantity)
    # Make sure this works with Dask and that these are still Dask arrays
    s_a = dask.array.from_array(s_a, chunks=s_a.shape)
    s_b = dask.array.from_array(s_b, chunks=s_b.shape)
    max_cc = max_cross_correlation(s_a, s_b, time, lag_bounds=bounds)
    tl = time_lag(s_a, s_b, time, lag_bounds=bounds)
    assert isinstance(max_cc, dask.array.Array)
    assert isinstance(tl, dask.array.Array)
