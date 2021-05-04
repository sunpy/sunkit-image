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
