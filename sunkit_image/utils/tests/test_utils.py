import warnings
from unittest.mock import patch

import numpy as np
import pytest

import astropy.io.fits
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

import sunpy.map

import sunkit_image.utils as utils
from sunkit_image import asda
from sunkit_image.data.test import get_test_filepath


def test_equally_spaced_bins():
    # test the default
    esb = utils.equally_spaced_bins()
    assert esb.shape == (2, 100)
    assert esb[0, 0] == 1.0
    assert esb[1, 0] == 1.01
    assert esb[0, 99] == 1.99
    assert esb[1, 99] == 2.00
    # Bins are 0.015 wide
    esb2 = utils.equally_spaced_bins(inner_value=0.5)
    assert esb2.shape == (2, 100)
    assert esb2[0, 0] == 0.5
    assert esb2[1, 0] == 0.515
    assert esb2[0, 99] == 1.985
    assert esb2[1, 99] == 2.00
    # Bins are 0.2 wide
    esb2 = utils.equally_spaced_bins(outer_value=3.0)
    assert esb2.shape == (2, 100)
    assert esb2[0, 0] == 1.0
    assert esb2[1, 0] == 1.02
    assert esb2[0, 99] == 2.98
    assert esb2[1, 99] == 3.00
    # Bins are 0.01 wide
    esb2 = utils.equally_spaced_bins(nbins=1000)
    assert esb2.shape == (2, 1000)
    assert esb2[0, 0] == 1.0
    assert esb2[1, 0] == 1.001
    assert esb2[0, 999] == 1.999
    assert esb2[1, 999] == 2.000
    # The radii have the correct relative sizes
    with pytest.raises(ValueError, match=r"The inner value must be strictly less than the outer value."):
        utils.equally_spaced_bins(inner_value=1.0, outer_value=1.0)
    with pytest.raises(ValueError, match=r"The inner value must be strictly less than the outer value."):
        utils.equally_spaced_bins(inner_value=1.5, outer_value=1.0)
    # The number of bins is strictly greater than 0
    with pytest.raises(ValueError, match=r"The number of bins must be strictly greater than 0."):
        utils.equally_spaced_bins(nbins=0)


def test_bin_edge_summary():
    esb = utils.equally_spaced_bins()
    center = utils.bin_edge_summary(esb, "center")
    assert center.shape == (100,)
    assert center[0] == 1.005
    assert center[99] == 1.995
    left = utils.bin_edge_summary(esb, "left")
    assert left.shape == (100,)
    assert left[0] == 1.0
    assert left[99] == 1.99
    right = utils.bin_edge_summary(esb, "right")
    assert right.shape == (100,)
    assert right[0] == 1.01
    assert right[99] == 2.0
    # Correct selection of summary type
    with pytest.raises(ValueError, match='Keyword "binfit" must have value "center", "left" or "right"'):
        utils.bin_edge_summary(esb, "should raise the error")
    # The correct shape of bin edges are passed in
    with pytest.raises(ValueError, match="The bin edges must be two-dimensional with shape \\(2, nbins\\)"):
        utils.bin_edge_summary(np.arange(0, 10), "center")
    with pytest.raises(ValueError, match="The bin edges must be two-dimensional with shape \\(2, nbins\\)"):
        utils.bin_edge_summary(np.zeros((3, 4)), "center")


@pytest.mark.remote_data()
def test_find_pixel_radii(aia_171):
    if isinstance(aia_171, np.ndarray):
        pytest.skip("This test is not compatible with numpy arrays")
    # The known maximum radius
    known_maximum_pixel_radius = 1.84183121
    # Calculate the pixel radii
    pixel_radii = utils.find_pixel_radii(aia_171)
    # The shape of the pixel radii is the same as the input map
    assert pixel_radii.shape[0] == int(aia_171.dimensions[0].value)
    assert pixel_radii.shape[1] == int(aia_171.dimensions[1].value)
    # Make sure the unit is solar radii
    assert pixel_radii.unit == u.R_sun
    # Make sure the maximum
    assert_quantity_allclose((np.max(pixel_radii)).value, known_maximum_pixel_radius)
    # Test that the new scale is used
    pixel_radii = utils.find_pixel_radii(aia_171, scale=2 * aia_171.rsun_obs)
    assert_quantity_allclose(np.max(pixel_radii).value, known_maximum_pixel_radius / 2)


@pytest.mark.remote_data()
def test_get_radial_intensity_summary(aia_171):
    if isinstance(aia_171, np.ndarray):
        pytest.skip("This test is not compatible with numpy arrays")
    radial_bin_edges = u.Quantity(utils.equally_spaced_bins(inner_value=1, outer_value=1.5)) * u.R_sun
    summary = np.mean
    map_r = utils.find_pixel_radii(aia_171, scale=aia_171.rsun_obs).to(u.R_sun)
    nbins = radial_bin_edges.shape[1]
    lower_edge = [map_r > radial_bin_edges[0, i].to(u.R_sun) for i in range(nbins)]
    upper_edge = [map_r < radial_bin_edges[1, i].to(u.R_sun) for i in range(nbins)]
    with warnings.catch_warnings():
        # We want to ignore RuntimeWarning: Mean of empty slice
        warnings.simplefilter("ignore", category=RuntimeWarning)
        expected = np.asarray([summary(aia_171.data[lower_edge[i] * upper_edge[i]]) for i in range(nbins)])
    assert np.allclose(utils.get_radial_intensity_summary(aia_171, radial_bin_edges=radial_bin_edges), expected)


def test_calculate_gamma():
    vel_file = get_test_filepath("asda_vxvy.fits")
    with astropy.io.fits.open(vel_file) as hdul:
        vxvy = {hdu.name.lower(): hdu.data for hdu in hdul[1:]}
    vx = vxvy["vx"]
    vy = vxvy["vy"]
    vxvy["data"]
    shape = vx.shape
    r = 3
    index = np.array([[i, j] for i in np.arange(r, shape[0] - r) for j in np.arange(r, shape[1] - r)])
    vel = asda.generate_velocity_field(vx, vy, index[1], index[0], r)
    pm = np.array(
        [[i, j] for i in np.arange(-r, r + 1) for j in np.arange(-r, r + 1)],
        dtype=float,
    )
    N = (2 * r + 1) ** 2
    pnorm = np.linalg.norm(pm, axis=1)
    cross = utils.utils._cross2d(pm, vel[..., 0])
    vel_norm = np.linalg.norm(vel[..., 0], axis=2)
    sint = cross / (pnorm * vel_norm + 1e-10)
    expected = np.nansum(sint, axis=1) / N
    assert np.allclose(expected, utils.calculate_gamma(pm, vel[..., 0], pnorm, N))


def test_remove_duplicate():
    rng = np.random.default_rng()
    test_data = rng.random(size=(5, 2))
    data_ = np.append(test_data, [test_data[0]], axis=0)
    expected = np.delete(data_, -1, 0)
    with pytest.raises(ValueError, match="Polygon must be defined as a n x 2 array!"):
        utils.remove_duplicate(data_.T)
    assert (utils.remove_duplicate(data_) == expected).all()


def test_points_in_poly():
    test_data = np.asarray([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 0]])
    with pytest.raises(ValueError, match="Polygon must be defined as a n x 2 array!"):
        utils.points_in_poly(test_data.T)
    expected = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    assert expected == utils.points_in_poly(test_data)


def test_reform_2d():
    test_data = np.asarray([[0, 0], [1, 2], [3, 4]])
    with pytest.raises(TypeError, match="Parameter 'factor' must be an integer!"):
        utils.reform2d(test_data, 2.2)
    with pytest.raises(ValueError, match="Input array must be 2d!"):
        utils.reform2d(test_data[0], 2)
    expected = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.75, 1.0, 1.0],
            [1.0, 1.5, 2.0, 2.0],
            [2.0, 2.5, 3.0, 3.0],
            [3.0, 3.5, 4.0, 4.0],
            [3.0, 3.5, 4.0, 4.0],
        ],
    )
    assert np.allclose(utils.reform2d(test_data, 2), expected)


# ---------------------------------------------------------------------------
# find_pixel_radii fast path + blackout_pixels_above_radius map_r reuse
# ---------------------------------------------------------------------------

# The synthetic ``_hpc_map`` fixtures below intentionally omit observer /
# obstime to stay network-free; sunpy emits informational metadata warnings
# in that case, which pytest's "warnings as errors" config promotes to
# failures.  Each test below silences these explicitly via per-test
# ``filterwarnings`` marks.


def _hpc_map(side=64, *, cdelt=50.0, crval=(0.0, 0.0), rotation=None):
    """Synthetic Helioprojective-Cartesian map with a known WCS.

    Skipping observer / obstime keeps the fixture network-free.  Pass a 2x2
    ``rotation`` to force the fast path's identity check to fail.
    """

    header = {
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "CTYPE1": "HPLN-TAN",
        "CTYPE2": "HPLT-TAN",
        "CDELT1": cdelt,
        "CDELT2": cdelt,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CRPIX1": (side + 1) / 2.0,
        "CRPIX2": (side + 1) / 2.0,
        "RSUN_REF": 6.957e8,
        "RSUN_OBS": 960.0,
    }
    if rotation is not None:
        header["PC1_1"], header["PC1_2"] = rotation[0]
        header["PC2_1"], header["PC2_2"] = rotation[1]
    data = np.random.default_rng(0).normal(size=(side, side))
    return sunpy.map.Map((data, header))


@pytest.mark.filterwarnings("ignore:Missing metadata for observer")
@pytest.mark.filterwarnings("ignore:Missing metadata for observation time")
@pytest.mark.filterwarnings("ignore:Missing metadata for solar radius")
def test_find_pixel_radii_fast_matches_slow():
    """Fast path on a non-rotated HPC map must match the SkyCoord-based slow
    path to better than the documented tolerance."""
    smap = _hpc_map(side=64)
    fast = utils.find_pixel_radii(smap).to(u.R_sun).value
    # Force the slow path: a tiny non-trivial rotation triggers the
    # identity check to fail and the SkyCoord path to run.
    rotated = _hpc_map(side=64, rotation=((0.99, 0.01), (-0.01, 0.99)))
    slow = utils.find_pixel_radii(rotated).to(u.R_sun).value
    # Equivalent fixtures up to a sub-degree rotation; check shape only here.
    assert fast.shape == slow.shape == (64, 64)
    # Stronger equivalence: same fixture, force slow path by mocking the
    # gate.  The values must be identical to within the floating-point
    # noise of the SkyCoord round-trip (the fast path mimics it).

    with patch("sunkit_image.utils.utils._is_simple_hpc", return_value=False):
        slow_same = utils.find_pixel_radii(smap).to(u.R_sun).value
    np.testing.assert_allclose(fast, slow_same, atol=2e-4)


@pytest.mark.filterwarnings("ignore:Missing metadata for observer")
@pytest.mark.filterwarnings("ignore:Missing metadata for observation time")
@pytest.mark.filterwarnings("ignore:Missing metadata for solar radius")
def test_find_pixel_radii_fast_off_center_sun():
    """``CRVAL != 0`` (off-center Sun) is still handled by the fast path."""

    smap = _hpc_map(side=64, crval=(200.0, -150.0))
    fast = utils.find_pixel_radii(smap).to(u.R_sun).value
    with patch("sunkit_image.utils.utils._is_simple_hpc", return_value=False):
        slow = utils.find_pixel_radii(smap).to(u.R_sun).value
    np.testing.assert_allclose(fast, slow, atol=2e-4)


@pytest.mark.filterwarnings("ignore:Missing metadata for observer")
@pytest.mark.filterwarnings("ignore:Missing metadata for observation time")
@pytest.mark.filterwarnings("ignore:Missing metadata for solar radius")
def test_find_pixel_radii_rotation_falls_through_to_slow_path():
    """A non-identity rotation matrix must drop through to the slow
    SkyCoord path."""

    rotated = _hpc_map(side=32, rotation=((0.9, 0.1), (-0.1, 0.9)))
    with patch("sunkit_image.utils.utils._find_pixel_radii_fast") as fast_mock:
        utils.find_pixel_radii(rotated)
    fast_mock.assert_not_called()


@pytest.mark.filterwarnings("ignore:Missing metadata for observer")
@pytest.mark.filterwarnings("ignore:Missing metadata for observation time")
@pytest.mark.filterwarnings("ignore:Missing metadata for solar radius")
def test_find_pixel_radii_fast_honors_scale_kwarg():
    """The ``scale`` argument must rescale the fast-path output the same way
    the slow path does."""
    smap = _hpc_map(side=32)
    default = utils.find_pixel_radii(smap).to(u.R_sun).value
    scaled = utils.find_pixel_radii(smap, scale=2 * smap.rsun_obs).to(u.R_sun).value
    np.testing.assert_allclose(scaled, default / 2.0, rtol=1e-10)


@pytest.mark.filterwarnings("ignore:Missing metadata for observer")
@pytest.mark.filterwarnings("ignore:Missing metadata for observation time")
@pytest.mark.filterwarnings("ignore:Missing metadata for solar radius")
def test_blackout_pixels_above_radius_map_r_reuse():
    """Supplying ``map_r=`` to ``blackout_pixels_above_radius`` must produce
    output identical to letting the function recompute it internally."""
    smap = _hpc_map(side=64)
    map_r = utils.find_pixel_radii(smap)
    blacked_internal = utils.blackout_pixels_above_radius(smap, 1.0 * u.R_sun).data
    blacked_reuse = utils.blackout_pixels_above_radius(smap, 1.0 * u.R_sun, map_r=map_r).data
    # NaN-aware byte equality
    finite_a = np.isfinite(blacked_internal)
    finite_b = np.isfinite(blacked_reuse)
    assert np.array_equal(finite_a, finite_b)
    np.testing.assert_array_equal(blacked_internal[finite_a], blacked_reuse[finite_b])


@pytest.mark.filterwarnings("ignore:Missing metadata for observer")
@pytest.mark.filterwarnings("ignore:Missing metadata for observation time")
@pytest.mark.filterwarnings("ignore:Missing metadata for solar radius")
def test_blackout_pixels_above_radius_map_r_skips_internal_lookup():
    """When ``map_r`` is supplied, ``blackout_pixels_above_radius`` must NOT
    call ``find_pixel_radii`` again (that's the whole point of the kwarg)."""

    smap = _hpc_map(side=32)
    map_r = utils.find_pixel_radii(smap)
    with patch("sunkit_image.utils.utils.find_pixel_radii") as mock_find:
        utils.blackout_pixels_above_radius(smap, 1.0 * u.R_sun, map_r=map_r)
    mock_find.assert_not_called()
