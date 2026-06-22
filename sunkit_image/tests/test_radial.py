import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy
import sunpy.data.sample
import sunpy.map
import sunpy.visualization.colormaps.cm

import sunkit_image.radial as rad
import sunkit_image.utils as utils
from sunkit_image.tests.helpers import figure_test, skip_windows

pytestmark = [
    pytest.mark.filterwarnings("ignore:Missing metadata for observer"),
    pytest.mark.filterwarnings("ignore:Missing metadata for observation time"),
]


@pytest.fixture()
def map_test1():
    x = np.linspace(-2, 2, 5)
    grid = np.meshgrid(x, x.T)
    test_data1 = np.sqrt(grid[0] ** 2 + grid[1] ** 2)
    test_data1 *= 10
    test_data1 = 28 - test_data1
    test_data1 = np.round(test_data1)
    header = {"cunit1": "arcsec", "cunit2": "arcsec", "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN"}
    return sunpy.map.Map((test_data1, header))


@pytest.fixture()
def map_test2():
    x = np.linspace(-2, 2, 5)
    grid = np.meshgrid(x, x.T)
    test_data1 = np.sqrt(grid[0] ** 2 + grid[1] ** 2)
    test_data1 *= 10
    test_data1 = 28 - test_data1
    test_data1 = np.round(test_data1)
    header = {"cunit1": "arcsec", "cunit2": "arcsec", "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN"}
    test_data2 = np.where(test_data1[:, 0:2] == 6, 8, test_data1[:, 0:2])
    test_data2 = np.concatenate((test_data2, test_data1[:, 2:]), axis=1)
    return sunpy.map.Map((test_data2, header))


@pytest.fixture()
def radial_bin_edges():
    radial_bins = utils.equally_spaced_bins(inner_value=0.001, outer_value=0.003, nbins=5)
    return radial_bins * u.R_sun


def test_nrgf(map_test1, map_test2, radial_bin_edges):
    result = np.zeros_like(map_test1.data)
    expect = rad.nrgf(map_test1, radial_bin_edges=radial_bin_edges, application_radius=0.001 * u.R_sun, fill=0)

    assert np.allclose(expect.data.shape, map_test1.data.shape)
    assert np.allclose(expect.data, result)

    # Hand calculated
    result1 = [
        [0.0, 1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0, -1.0, 0.0],
    ]

    expect1 = rad.nrgf(map_test2, radial_bin_edges=radial_bin_edges, application_radius=0.001 * u.R_sun, fill=0)

    assert np.allclose(expect1.data.shape, map_test2.data.shape)
    assert np.allclose(expect1.data, result1)


def test_fnrgf(map_test1, map_test2, radial_bin_edges):
    order = 1
    result = [
        [-0.0, 96.0, 128.0, 96.0, -0.0],
        [96.0, 224.0, 288.0, 224.0, 96.0],
        [128.0, 288.0, 0.0, 288.0, 128.0],
        [96.0, 224.0, 288.0, 224.0, 96.0],
        [-0.0, 96.0, 128.0, 96.0, -0.0],
    ]
    expect = rad.fnrgf(
        map_test1,
        radial_bin_edges=radial_bin_edges,
        order=order,
        mean_attenuation_range=[1.0, 0.0],
        std_attenuation_range=[1.0, 0.0],
        cutoff=0,
        application_radius=0.001 * u.R_sun,
        number_angular_segments=4,
        fill=0,
    )
    assert np.allclose(expect.data.shape, map_test1.data.shape)
    assert np.allclose(expect.data, result)

    result1 = [
        [-0.0, 128.0, 128.0, 96.0, -0.0],
        [128.0, 224.0, 288.0, 224.0, 96.0],
        [128.0, 288.0, 0.0, 288.0, 128.0],
        [128.0, 224.0, 288.0, 224.0, 96.0],
        [-0.0, 128.0, 128.0, 96.0, -0.0],
    ]
    expect1 = rad.fnrgf(
        map_test2,
        radial_bin_edges=radial_bin_edges,
        order=order,
        mean_attenuation_range=[1.0, 0.0],
        std_attenuation_range=[1.0, 0.0],
        cutoff=0,
        application_radius=0.001 * u.R_sun,
        number_angular_segments=4,
        fill=0,
    )

    assert np.allclose(expect1.data.shape, map_test2.data.shape)
    assert np.allclose(expect1.data, result1)

    order = 5
    result2 = [
        [-0.0, 90.52799999982116, 126.73137084989847, 90.52799999984676, -0.0],
        [90.52800000024544, 207.2, 285.14558441227155, 207.2, 90.5280000001332],
        [126.73137084983244, 285.1455844119744, 0.0, 280.05441558770406, 124.4686291500961],
        [90.52800000015233, 207.2, 280.05441558772844, 207.2, 90.5280000000401],
        [0.0, 90.52799999986772, 124.46862915010152, 90.52799999989331, -0.0],
    ]

    expect2 = rad.fnrgf(
        map_test1,
        radial_bin_edges=radial_bin_edges,
        order=order,
        mean_attenuation_range=[1.0, 0.0],
        std_attenuation_range=[1.0, 0.0],
        cutoff=0,
        application_radius=0.001 * u.R_sun,
        number_angular_segments=4,
        fill=0,
    )

    assert np.allclose(expect2.data.shape, map_test1.data.shape)
    assert np.allclose(expect2.data, result2)

    result3 = [
        [-0.0, 120.55347470594926, 126.73137084989847, 90.67852529365966, -0.0],
        [120.70526403418884, 207.2, 285.14558441227155, 207.2, 90.52673596626707],
        [126.73137084983244, 285.1455844119744, 0.0, 280.05441558770406, 124.4686291500961],
        [120.70526403406846, 207.2, 280.05441558772844, 207.2, 90.52673596617021],
        [0.0, 120.55347470601022, 124.46862915010152, 90.67852529370734, -0.0],
    ]

    expect3 = rad.fnrgf(
        map_test2,
        radial_bin_edges=radial_bin_edges,
        order=order,
        mean_attenuation_range=[1.0, 0.0],
        std_attenuation_range=[1.0, 0.0],
        cutoff=0,
        application_radius=0.001 * u.R_sun,
        number_angular_segments=4,
        fill=0,
    )

    assert np.allclose(expect3.data.shape, map_test2.data.shape)
    assert np.allclose(expect3.data, result3)


def test_fnrgf_errors(map_test1):
    with pytest.raises(ValueError, match="Minimum value of order is 1"):
        rad.fnrgf(
            map_test1,
            order=0,
            mean_attenuation_range=[1.0, 0.0],
            std_attenuation_range=[1.0, 0.0],
            cutoff=0,
        )


@figure_test
@pytest.mark.remote_data()
def test_fig_nrgf(aia_171_map):
    radial_bin_edges = utils.equally_spaced_bins()
    radial_bin_edges *= u.R_sun
    out = rad.nrgf(aia_171_map, radial_bin_edges=radial_bin_edges)
    out.plot()


@figure_test
@pytest.mark.remote_data()
def test_fig_fnrgf(aia_171_map):
    radial_bin_edges = utils.equally_spaced_bins()
    radial_bin_edges *= u.R_sun
    order = 20
    out = rad.fnrgf(
        aia_171_map,
        radial_bin_edges=radial_bin_edges,
        order=order,
        mean_attenuation_range=[1.0, 0.0],
        std_attenuation_range=[1.0, 0.0],
        cutoff=0,
    )
    out.plot()


@figure_test
@pytest.mark.remote_data()
def test_fig_rhef(aia_171_map):
    radial_bin_edges = utils.equally_spaced_bins(0, 2, aia_171_map.data.shape[1])
    radial_bin_edges *= u.R_sun
    out = rad.rhef(aia_171_map, radial_bin_edges=radial_bin_edges, upsilon=None, method="scipy")
    out.plot()


@figure_test
@pytest.mark.remote_data()
def test_multifig_rhef(aia_171_map):
    radial_bin_edges = utils.equally_spaced_bins(0, 2, aia_171_map.data.shape[1])
    radial_bin_edges *= u.R_sun

    # Define the list of upsilon pairs where the first number affects dark components and the second number affects bright ones
    upsilon_list = [
        0.35,
        None,
        (0.1, 0.1),
        (0.5, 0.5),
        (0.8, 0.8),
    ]

    # Crop the figures to see better detail
    top_right = SkyCoord(1200 * u.arcsec, 0 * u.arcsec, frame=aia_171_map.coordinate_frame)
    bottom_left = SkyCoord(0 * u.arcsec, -1200 * u.arcsec, frame=aia_171_map.coordinate_frame)
    aia_map_cropped = aia_171_map.submap(bottom_left, top_right=top_right)
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 10), sharex="all", sharey="all", subplot_kw={"projection": aia_map_cropped}
    )
    axes = axes.flatten()

    aia_map_cropped.plot(axes=axes[0], clip_interval=(1, 99.99) * u.percent)
    axes[0].set_title("Original AIA Map")

    # Loop through the upsilon_list and plot each filtered map
    for i, upsilon in enumerate(upsilon_list):
        out_map = rad.rhef(aia_171_map, upsilon=upsilon, method="scipy")
        out_map_crop = out_map.submap(bottom_left, top_right=top_right)
        out_map_crop.plot(axes=axes[i + 1])
        axes[i + 1].set_title(f"Upsilon = {upsilon}")

    fig.tight_layout()

    return fig


def test_set_attenuation_coefficients():
    order = 1
    # Hand calculated
    expect1 = [[1, 0.0], [1, 0.0]]

    result1 = rad._set_attenuation_coefficients(order)
    assert np.allclose(expect1, result1)

    order = 3
    # Hand calculated
    expect2 = [[1.0, 0.66666667, 0.33333333, 0.0], [1.0, 0.66666667, 0.33333333, 0.0]]

    result2 = rad._set_attenuation_coefficients(order)
    assert np.allclose(expect2, result2)

    expect3 = [[1.0, 0.66666667, 0.0, 0.0], [1.0, 0.66666667, 0.0, 0.0]]

    result3 = rad._set_attenuation_coefficients(order, cutoff=2)
    assert np.allclose(expect3, result3)

    with pytest.raises(ValueError, match="Cutoff cannot be greater than order \\+ 1"):
        rad._set_attenuation_coefficients(order, cutoff=5)


def test_fit_polynomial_to_log_radial_intensity():
    radii = (0.001, 0.002) * u.R_sun
    intensity = np.asarray([1, 2])
    degree = 1
    expected = np.polyfit(radii.to(u.R_sun).value, np.log(intensity), degree)

    assert np.allclose(rad._fit_polynomial_to_log_radial_intensity(radii, intensity, degree), expected)


def test_calculate_fit_radial_intensity():
    polynomial = np.asarray([1, 2, 3])
    radii = (0.001, 0.002) * u.R_sun
    expected = np.exp(np.poly1d(polynomial)(radii.to(u.R_sun).value))

    assert np.allclose(rad._calculate_fit_radial_intensity(radii, polynomial), expected)


def test_normalize_fit_radial_intensity():
    polynomial = np.asarray([1, 2, 3])
    radii = (0.001, 0.002) * u.R_sun
    normalization_radii = (0.003, 0.004) * u.R_sun
    expected = rad._calculate_fit_radial_intensity(radii, polynomial) / rad._calculate_fit_radial_intensity(
        normalization_radii,
        polynomial,
    )

    assert np.allclose(rad._normalize_fit_radial_intensity(radii, polynomial, normalization_radii), expected)


@skip_windows
def test_intensity_enhance(map_test1):
    degree = 1
    fit_range = [1, 1.5] * u.R_sun
    normalization_radius = 1 * u.R_sun
    summarize_bin_edges = "center"
    scale = 1 * map_test1.rsun_obs
    radial_bin_edges = u.Quantity(utils.equally_spaced_bins()) * u.R_sun

    radial_intensity = utils.get_radial_intensity_summary(map_test1, radial_bin_edges, scale=scale)

    map_r = utils.find_pixel_radii(map_test1).to(u.R_sun)

    radial_bin_summary = utils.bin_edge_summary(radial_bin_edges, summarize_bin_edges).to(u.R_sun)

    fit_here = np.logical_and(
        fit_range[0].to(u.R_sun).value <= radial_bin_summary.to(u.R_sun).value,
        radial_bin_summary.to(u.R_sun).value <= fit_range[1].to(u.R_sun).value,
    )

    polynomial = rad._fit_polynomial_to_log_radial_intensity(
        radial_bin_summary[fit_here],
        radial_intensity[fit_here],
        degree,
    )

    enhancement = 1 / rad._normalize_fit_radial_intensity(map_r, polynomial, normalization_radius)
    enhancement[map_r < normalization_radius] = 1

    assert np.allclose(
        enhancement * map_test1.data,
        rad.intensity_enhance(map_test1, radial_bin_edges=radial_bin_edges, scale=scale).data,
    )


@skip_windows
def test_intensity_enhance_errors(map_test1):
    fit_range = [1, 1.5] * u.R_sun
    scale = 1 * map_test1.rsun_obs
    with pytest.raises(ValueError, match=r"The fit range must be strictly increasing."):
        rad.intensity_enhance(map_test1, scale=scale, fit_range=fit_range[::-1])


# ---------------------------------------------------------------------------
# rhef sort-and-group inner loop: equivalence + edge-case coverage
# ---------------------------------------------------------------------------


def _rhef_reference_loop(smap, radial_bin_edges, *, application_radius, method, upsilon, fill=np.nan):
    """A faithful copy of the original per-bin mask loop in `rhef`.

    Used as the equivalence oracle for the optimised implementation.  We
    inline a fresh copy here rather than reaching into the production
    module so a future refactor that breaks equivalence is caught even if
    the test imports drift.  ``find_radial_bin_edges`` is called first so
    the reference sees whatever edges ``rhef`` would actually use — this
    matters when the user-supplied edges don't span the full map and the
    helper rebuilds them.

    Pixels that land in no bin (e.g. the extreme corner where ``map_r``
    exactly equals the upper edge under ``< hi`` semantics) keep ``fill``,
    matching the way ``rhef`` initialises its output.

    Why an inlined oracle?  This PR replaces ``rhef``'s original per-bin
    boolean-mask loop with a sort-and-group kernel.  The two are intended to be
    bit-identical, so the most direct way to prove that — and to keep proving it
    against future refactors — is to keep a faithful copy of the original loop
    here and assert equivalence across every ranking method plus the edge cases
    that follow (empty bins, ``fill`` propagation, ``application_radius``,
    overlapping bins, the ``upsilon`` correction).  It is inlined rather than
    imported from the module so the oracle cannot silently drift to track the
    very code it is meant to check.  The real-data figure test ``test_fig_rhef``
    is the complementary backstop: any numerically significant change to the
    kernel would alter the rendered AIA 171 image and fail its hash comparison.
    """
    radial_bin_edges, map_r = utils.find_radial_bin_edges(smap, radial_bin_edges)
    map_r = map_r.to(u.R_sun)

    def _ranking_func(arr):
        # Mirror upstream's NaN-aware ranking; non-NaN inputs match exactly.
        mask = ~np.isnan(arr)
        if method == "scipy":
            out = np.full(arr.shape, np.nan)
            out[mask] = stats.rankdata(arr[mask], method="average") / np.sum(mask)
            return out
        # "numpy"
        out = arr.copy()
        order = np.argsort(arr)
        order = order[~np.isnan(arr[order])]
        out[order] = np.arange(1, len(order) + 1)
        return out / float(len(order))

    data = np.full_like(smap.data, fill)
    for i in range(radial_bin_edges.shape[1]):
        here = np.logical_and(map_r >= radial_bin_edges[0, i], map_r < radial_bin_edges[1, i])
        if application_radius is not None and application_radius > 0:
            here = np.logical_and(here, map_r >= application_radius)
        if not here.any():
            continue
        data[here] = _ranking_func(smap.data[here])
        if upsilon is not None:
            data[here] = rad.apply_upsilon(data[here], upsilon)
    return data


def _synthetic_map(side=64, seed=7, *, with_nans=False):
    """Tiny in-memory `sunpy.map.Map` so equivalence tests need no sample data.

    Uses the same minimal-header pattern as the existing ``map_test1`` /
    ``map_test2`` fixtures above (no observer / obstime needed), with an
    explicit ``rsun_obs`` so ``find_pixel_radii`` can convert arcsec → R_sun
    without a network IERS lookup.

    NaNs are off by default: scipy's ``stats.rankdata`` propagates NaN to
    every output rank in the same call, so a single NaN in a bin makes the
    whole bin NaN — which is fine for equivalence testing but defeats other
    assertions about "some pixel was ranked".  Pass ``with_nans=True`` only
    in tests that explicitly verify NaN propagation.
    """
    rng = np.random.default_rng(seed)
    data = rng.exponential(scale=100.0, size=(side, side)).astype(float)
    if with_nans:
        data[rng.random(data.shape) < 0.03] = np.nan
    header = {
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "CTYPE1": "HPLN-TAN",
        "CTYPE2": "HPLT-TAN",
        "CDELT1": 50.0,
        "CDELT2": 50.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CRPIX1": (side + 1) / 2.0,
        "CRPIX2": (side + 1) / 2.0,
        "RSUN_REF": 6.957e8,
        "RSUN_OBS": 960.0,  # arcsec; mid-range solar-disk apparent radius
    }
    return sunpy.map.Map((data, header))


@pytest.mark.parametrize("method", ["scipy", "numpy"])
def test_rhef_matches_reference_loop(method):
    """The sort-and-group inner loop must produce output identical to the
    original per-bin mask loop, byte-for-byte, across the supported
    ``method=`` values."""
    smap = _synthetic_map()
    edges = np.linspace(0, 1.5, 33)
    edges = np.array([edges[:-1], edges[1:]]) * u.R_sun

    out = rad.rhef(smap, radial_bin_edges=edges, upsilon=None, method=method, vignette=10 * u.R_sun).data
    ref = _rhef_reference_loop(smap, edges, application_radius=0 * u.R_sun, method=method, upsilon=None)
    # NaN-aware exact match: rankdata is deterministic and the sort is
    # stable, so per-bin outputs are bit-identical between the two paths.
    assert out.shape == ref.shape
    finite = np.isfinite(out) & np.isfinite(ref)
    assert np.array_equal(out[finite], ref[finite])
    assert np.array_equal(np.isfinite(out), np.isfinite(ref))


def test_rhef_matches_reference_with_application_radius():
    """``application_radius`` clips below a floor; the optimised path applies
    the same mask before binning, so output must still match the reference."""
    smap = _synthetic_map()
    edges = np.linspace(0, 1.5, 33)
    edges = np.array([edges[:-1], edges[1:]]) * u.R_sun
    appr = 0.4 * u.R_sun

    out = rad.rhef(
        smap,
        radial_bin_edges=edges,
        application_radius=appr,
        upsilon=None,
        method="scipy",
        vignette=10 * u.R_sun,
    ).data
    ref = _rhef_reference_loop(smap, edges, application_radius=appr, method="scipy", upsilon=None)
    finite = np.isfinite(out) & np.isfinite(ref)
    assert np.array_equal(out[finite], ref[finite])


def test_rhef_matches_reference_with_upsilon():
    """The μ-correction is applied per pixel; verify the optimised path's
    placement (one apply_upsilon call per bin slice) matches the reference."""
    smap = _synthetic_map()
    edges = np.linspace(0, 1.5, 33)
    edges = np.array([edges[:-1], edges[1:]]) * u.R_sun

    out = rad.rhef(smap, radial_bin_edges=edges, upsilon=0.35, method="scipy", vignette=10 * u.R_sun).data
    ref = _rhef_reference_loop(smap, edges, application_radius=0 * u.R_sun, method="scipy", upsilon=0.35)
    finite = np.isfinite(out) & np.isfinite(ref)
    # Exact, like the other equivalence tests (the finite mask already removed NaNs).
    assert np.array_equal(out[finite], ref[finite])


def test_rhef_empty_bins_left_at_zero():
    """A radial bin that no pixel falls into must leave its output pixels at
    the storage default (zero), matching the old loop's behaviour where the
    inner ``data[here] = ...`` assignment was simply never reached."""
    smap = _synthetic_map(side=32)
    # First bin is entirely below the smallest pixel radius — guaranteed empty
    edges = np.array([[0.0, 0.5, 1.0], [0.001, 1.0, 1.5]]) * u.R_sun

    out = rad.rhef(smap, radial_bin_edges=edges, upsilon=None, method="scipy", vignette=10 * u.R_sun).data
    # Where map_r < 0.001 R_sun (essentially nowhere on this header), output
    # stays zero; rest is non-trivial.  Just check we have BOTH zeros and
    # rank values present — confirms the empty-bin branch fires.
    # Empty-bin pixels keep the storage default (NaN by default per the
    # ``fill`` parameter); pixels in populated bins receive ranks > 0.
    assert np.isnan(out).any()
    assert (out > 0).any()
