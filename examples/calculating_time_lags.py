"""
==========================================
Computing Cross-Correlations and Time Lags
==========================================

This example shows how to compute cross-correlations
between light curves and map the resulting time lags,
those temporal offsets which maximize the cross-correlation
between the two signals, back to an image pixel.
This method
was developed for studying temporal evolution of AIA intensities
by `Viall and Klimchuk (2012) <https://doi.org/10.1088/0004-637X/753/1/35>`_.
The specific implementation in this package is described in detail
in Appendix C of `Barnes et al. (2019) <https://doi.org/10.3847/1538-4357/ab290c>`_.
"""
# sphinx_gallery_thumbnail_number = 4
import dask.array
import matplotlib.pyplot as plt
import numpy as np

from sunkit_image.time_lag import cross_correlation, get_lags, max_cross_correlation, time_lag


###################################################################
# Consider two timeseries whose peaks are separated in time by some
# interval. We will create a toy model with two Gaussian pulses. In
# practice, this method is often applied to many AIA light curves
# and is described in detail in
# `Viall and Klimchuk (2012) <https://doi.org/10.1088/0004-637X/753/1/35>`_.
def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


time = np.linspace(0, 1, 500)
s_a = gaussian_pulse(time, 0.4, 0.02)
s_b = gaussian_pulse(time, 0.6, 0.02)
plt.plot(time, s_a)
plt.plot(time, s_b)

###################################################################
# The problem we are concerned with is how much do we need shift
# signal A, either forward or backward in time, in order for it to
# best line up with signal B. In other words, what is the time lag
# between A and B. In the context of analyzing light curves from
# AIA, this gives us a proxy for the cooling time between two
# narrowband channels and thus two temperatures. To find this,
# we can compute the cross-correlation between the two signals
# and find which "lag" yields the highest correlation.
lags = get_lags(time)
cc = cross_correlation(s_a, s_b, lags)
plt.plot(lags, cc)
plt.show()

###################################################################
# Additionally, we can also easily calculate the maximum value of the
# cross-correlation and the associate lag, or the time lag.
tl = time_lag(s_a, s_b, time)
max_cc = max_cross_correlation(s_a, s_b, time)
plt.plot(lags, cc)
plt.plot(tl, max_cc, marker="o", ls="", markersize=4)
plt.show()

###################################################################
# As expected from the first intensity plot, we find that the lag
# which maximizes the cross-correlation is approximately the separation
# between the mean values of the Gaussian pulses.
print("Time lag, A->B = ", tl)

###################################################################
# Note that a positive time lag indicates that signal A has to be
# shifted forward in time to match signal B. By reversing the order
# of the inputs, we also reverse the sign of the time lag.
print("Time lag, B->A =", time_lag(s_b, s_a, time))

###################################################################
# The real power in the time lag approach is it's ability to reveal
# large scale patterns of cooling in images of the Sun, particularly
# in active regions. All of these functions can also be applied to
# intensity data cubes to create a "time lag map".
#
# As an example, we'll create a fake data cube by repeating Gaussian
# pulses with varying means and then add some noise to them
time = np.tile(time, (10, 10, 1)).T
means_a = np.tile(np.random.rand(*time.shape[1:]), (time.shape[0], 1, 1))
means_b = np.tile(np.random.rand(*time.shape[1:]), (time.shape[0], 1, 1))
noise = -0.05 + 0.1 * np.random.rand(*means_a.shape)
s_a = gaussian_pulse(time, means_a, 0.02) + noise
s_b = gaussian_pulse(time, means_b, 0.02) + noise

###################################################################
# We can now compute a map of the time lag and maximum cross correlation
max_cc_map = max_cross_correlation(s_a, s_b, time[:, 0, 0])
tl_map = time_lag(s_a, s_b, time[:, 0, 0])
plt.subplot(121)
im = plt.imshow(tl_map, cmap="RdBu", vmin=-1, vmax=1)
plt.colorbar(im)
plt.subplot(122)
im = plt.imshow(max_cc_map, vmin=0, vmax=1)
plt.colorbar(im)
plt.show()

###################################################################
# In practice, these data cubes are often very large, sometimes many
# GB, such that doing operations like these on them can be prohibitively
# expensive. All of these operations can be parallelized and distributed
# easily by passing in the intensity cubes as Dask arrays.
s_a = dask.array.from_array(s_a, chunks=s_a.shape[:1] + (5, 5))
s_b = dask.array.from_array(s_b, chunks=s_b.shape[:1] + (5, 5))
time = dask.array.from_array(time[:, 0, 0], chunks=time.shape[:1])
tl_map = time_lag(s_a, s_b, time)
print(tl_map)

###################################################################
# Rather than being computed "eagerly", :func:`time_lag` returns
# a graph of the computation that can be handed off to a distributed
# scheduler to be run in parallel. This is extremely advantageous for
# large data cubes as these operations are likely to exceed the
# memory limits of most desktop machines are easily accelerated through
# parallelism.
