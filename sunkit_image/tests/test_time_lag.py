import dask.array
import numpy as np
import pytest

from sunkit_image.time_lag import cross_correlation, get_lags, max_cross_correlation, time_lag


@pytest.mark.parametrize(
    "shape_in,shape_out", [((20, 5, 5), (39, 5, 5)), ((100, 10), (199, 10)), ((1000,), (1999,))]
)
def test_cross_correlation_array_shapes(shape_in, shape_out):
    s_a = np.random.rand(*shape_in)
    s_b = np.random.rand(*shape_in)
    time = np.linspace(0, 1, shape_in[0])
    lags = get_lags(time)
    cc = cross_correlation(s_a, s_b, lags)
    assert cc.shape == shape_out


@pytest.mark.parametrize("shape", [((5, 5)), ((10,)), ((1,))])
def test_max_cc_time_lag_array_shapes(shape):
    time = np.linspace(0, 1, 10)
    shape_in = time.shape + shape
    s_a = np.random.rand(*shape_in)
    s_b = np.random.rand(*shape_in)
    max_cc = max_cross_correlation(s_a, s_b, time)
    tl = time_lag(s_a, s_b, time)
    assert max_cc.shape == shape
    assert tl.shape == shape


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
    time = np.linspace(0, 1, shape_in[0])
    # Numpy arrays
    max_cc = max_cross_correlation(s_a, s_b, time)
    tl = time_lag(s_a, s_b, time)
    assert isinstance(max_cc, np.ndarray)
    assert isinstance(tl, np.ndarray)
    # Dask arrays
    s_a = dask.array.from_array(s_a)
    s_b = dask.array.from_array(s_b)
    time = dask.array.from_array(time)
    max_cc = max_cross_correlation(s_a, s_b, time)
    tl = time_lag(s_a, s_b, time)
    assert isinstance(max_cc, dask.array.Array)
    assert isinstance(tl, dask.array.Array)
