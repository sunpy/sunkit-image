0.5.1 (2023-11-17)
==================

Trivial/Internal Changes
------------------------

- Added the explicitly imported packages to the install requirements. (`#160 <https://github.com/sunpy/sunkit-image/pull/160>`__)

0.5.0 (2023-08-10)
==================

Backwards Incompatible Changes
------------------------------

- The following helper functions in `sunkit_image.colaginment` have been removed, with no replacement.
  This is because they are designed to be internal helper functions.
  If you need to use them in your own code create a copy of the functions from the ``sunkit-image`` source code.

  - ``parabolic_turning_point``
  - ``calculate_clipping``
  - ``check_for_nonfinite_entries``
  - ``get_correlation_shifts``
  - ``clip_edges``
  - ``find_best_match_location``
  - ``calculate_shift`` (`#100 <https://github.com/sunpy/sunkit-image/pull/100>`__)

- The following helper functions in `sunkit_image.radial` have been removed, with no replacement.
  This is because they are designed to be internal helper functions.
  If you need to use them in your own code create a copy of the functions from the ``sunkit-image`` source code.

  - ``fit_polynomial_to_log_radial_intensity``
  - ``calculate_fit_radial_intensity``
  - ``normalize_fit_radial_intensity``

- Made the following functions in `sunkit_image.trace` private:

  1. ``curvature_radius`` (renamed to ``_curvature_radius``)
  2. ``erase_loop_in_image`` (renamed to ``_erase_loop_in_image``)
  3. ``initial_direction_finding`` (renamed to ``_initial_direction_finding``)
  4. ``loop_add`` (renamed to ``_loop_add``)

  These were never intended to be used by users but for the user-facing functions. (`#136 <https://github.com/sunpy/sunkit-image/pull/136>`__)

- Dropped support for Python 3.8 by increasing minimum required Python version to 3.9.
  Dropped support for sunpy 4.0 and 4.1 by increasing minimum required sunpy version to 5.0.
  Dropped support for scikit-image 0.18 by increasing minimum required scikit-image version to 0.19. (`#155 <https://github.com/sunpy/sunkit-image/pull/155>`__)


Features
--------

- Add two examples demonstrating the usage of :func:`~sunkit_image.coalignment.mapsequence_coalign_by_match_template` and :func:`~sunkit_image.coalignment.mapsequence_coalign_by_rotation`. (`#90 <https://github.com/sunpy/sunkit-image/pull/90>`__)
- Added the `sunkit_image.granule` module which provides functions to segment granulation in images of the solar photosphere.
  The key functionality is contained in the `~sunkit_image.granule.segment` function, which
  segments an image into intergranule, granule, faculae, and, optionally, dim granule. (`#114 <https://github.com/sunpy/sunkit-image/pull/114>`__)
- ``mypy`` type checking has been enabled on the repository.
  Types have not yet been extensively added, but running ``mypy`` does not raise any errors. (`#133 <https://github.com/sunpy/sunkit-image/pull/133>`__)
- Several functions have been updated to accept either numpy array or sunpy map inputs.
  The following functions now accept either a numpy array or sunpy map, and return the same data type:

  - `sunkit_image.enhance.mgn`
  - `sunkit_image.trace.bandpass_filter`
  - `sunkit_image.trace.smooth`

  The following functions now accept either a numpy array or sunpy map, and their return type is unchanged:

  - `sunkit_image.trace.occult2` (`#135 <https://github.com/sunpy/sunkit-image/pull/135>`__)
- Modifications to the `sunkit_image.granule` module.

  1. Increase in speed for large images achieved by computing the initial thresholding on a random subset of pixels.
  2. Increase accuracy on images with spatially varying background flux levels achieved by applying a local histogram equalization before computing the initial thresholding.
  3. Prevention of errors in finding "dim centers" in images that have all-granule edges achieved by adding a "padding" of zero pixels around the edges.
  4. Correction of the assignment of the values 2 and 3 to brightpoints and dim centers. (`#154 <https://github.com/sunpy/sunkit-image/pull/154>`__)


Improved Documentation
----------------------

- Added two notes to `sunkit_image.enhance.mgn` detailing prerequisites for using this function. (`#126 <https://github.com/sunpy/sunkit-image/pull/126>`__)
- Added a tutorial (:ref:`sphx_glr_generated_gallery_rgb_composite.py`) demonstrating how to create an RGB image with three different maps. (`#128 <https://github.com/sunpy/sunkit-image/pull/128>`__)


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
