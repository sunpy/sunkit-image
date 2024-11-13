import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy
import sunpy.data.sample
import sunpy.map
import sunpy.visualization.colormaps.cm

import sunkit_image.radial as rad
import sunkit_image.utils as utils
from sunkit_image.tests.helpers import figure_test, skip_windows

pytestmark = [pytest.mark.filterwarnings("ignore:Missing metadata for observer"), pytest.mark.filterwarnings("ignore:Missing metadata for observation time")]


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
    out = rad.fnrgf(aia_171_map, radial_bin_edges=radial_bin_edges, order=order, mean_attenuation_range=[1.0, 0.0], std_attenuation_range=[1.0, 0.0], cutoff=0)
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
    with pytest.raises(ValueError, match="The fit range must be strictly increasing."):
        rad.intensity_enhance(map_test1, scale=scale, fit_range=fit_range[::-1])
