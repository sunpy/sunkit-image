0.4.2 (2022-05-24)
==================

Breaking Changes
----------------

- Minimum version of ``sunpy`` required is now 4.0.0

0.4.1 (2022-04-05)
==================

Features
--------

- Add `~sunkit_image.coalignment.calculate_solar_rotate_shift` and
  `~sunkit_image.coalignment.mapsequence_solar_derotate` to
  the `sunkit_image.coalignment` module. (`#81 <https://github.com/sunpy/sunkit-image/pull/81>`__)


0.4.0 (2022-03-11)
==================

Features
--------

- Add the `~sunkit_image.coalignment` module ported from `~sunpy.image.coalignment`. (`#78 <https://github.com/sunpy/sunkit-image/pull/78>`__)


0.3.2 (2022-03-08)
==================

Trivial/Internal Changes
------------------------

- Minor changes to ensure that sunkit-image is buildable on conda-forge.

0.3.1 (2021-11-19)
==================

- Fixed a bug where a `~astropy.units.UnitConversionError` was thrown if a non-dimensionless
  `~astropy.units.Quantity` object was input for the signal in `~sunkit_image.time_lag.cross_correlation`. (`#72 <https://github.com/sunpy/sunkit-image/pull/72>`__)
- Fixed a bug where the way we dealt with `~astropy.unit.Quantity` objects was inconsistent with
  `~dask.array.Array` objects in newer versions of `~numpy`. The `pre_check_hook` option keyword
  argument has also been removed from `~sunkit_image.time_lag.time_lag` and `post_check_hook`
  has been renamed to `array_check` and now accepts two arguments. (`#72 <https://github.com/sunpy/sunkit-image/pull/72>`__)


Trivial/Internal Changes
------------------------

- A warning is now raised if the input data to `~sunkit_image.enhance.mgn` contain any NaNs. (`#73 <https://github.com/sunpy/sunkit-image/pull/73>`__)

0.3.0 (2021-06-02)
==================

Features
--------

- The `sunkit_image.time_lag` module provides functions for computing the cross-correlation,
  time lag, and peak cross-correlation for N-dimensional time series. (`#61 <https://github.com/sunpy/sunkit-image/pull/61>`__)
- Increased the minimum version of "sunpy" to 3.0.0, the new LTS release

0.2.0 (2021-05-04)
==================

Features
--------

- The minimum and maximum values of the gamma transform can now be specified for :func:`sunkit_image.enhance.mgn`. (`#60 <https://github.com/sunpy/sunkit-image/pull/60>`__)


Bug Fixes
---------

- Increased the minimum version of "skimage" to 0.18.0, preventing faulty code in :meth:`sunkit-image.utils.points_in_poly`. (`#59 <https://github.com/sunpy/sunkit-image/pull/59>`__)


Trivial/Internal Changes
------------------------

- Added multiple unit tests to increase code coverage. (`#59 <https://github.com/sunpy/sunkit-image/pull/59>`__)
- Increased minimum supported version of sunpy to 2.0.0
- Many internal package updates to documentation, the continuous integration and etc.

0.1.0 (2020-04-30)
==================

Features
--------

- Added a class (`sunkit_image.utils.noise.NoiseLevelEstimation`) for noise level estimation of an image. (`#12 <https://github.com/sunpy/sunkit-image/pull/12>`__)
- Added a new function (`sunkit_image.radial.fnrgf`) to normalize the radial brightness gradient using a Fourier approximation. (`#17 <https://github.com/sunpy/sunkit-image/pull/17>`__)
- Added a function (`sunkit_image.enhance.mgn`) for applying Multi-scale Gaussian Normalization to an image (`numpy.ndarray`). (`#30 <https://github.com/sunpy/sunkit-image/pull/30>`__)
- Added a new function (`sunkit_image.trace.occult2`) to automatically trace out loops/curved structures in an image. (`#31 <https://github.com/sunpy/sunkit-image/pull/31>`__)
- Added an implementation of the Automated Swirl Detection Algorithm (ASDA). (`#40 <https://github.com/sunpy/sunkit-image/pull/40>`__)


Improved Documentation
----------------------

- Added an example on how to use `astroscrappy.detect_cosmics <https://astroscrappy.readthedocs.io/en/latest/api/astroscrappy.detect_cosmics.html>`__ to eliminate cosmic ray hits in solar images. (`#35 <https://github.com/sunpy/sunkit-image/pull/35>`__)


Trivial/Internal Changes
------------------------

- Transferred sunkit_image.utils.noise.NoiseLevelEstimation from class object into a series of functions. (`#38 <https://github.com/sunpy/sunkit-image/pull/38>`__)
