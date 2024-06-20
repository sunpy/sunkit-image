import numpy as np
import pytest

from sunkit_image import asda
from sunkit_image.data.test import get_test_filepath


def test_asda_artificial():
    """
    Generate an artificial vortex using the Lamb_Oseen class in asda, then
    perform the vortex detection.
    """
    # Generate an artificial vortex
    vmax = 2.0  # rotating speed
    rmax = 50  # radius
    ratio = 0.2  # ratio of expanding speed over rotating speed
    alpha = 1.256430
    rcore = rmax / np.sqrt(alpha)
    gamma = 2 * np.pi * vmax * rmax * (1 + 1 / (2 * alpha))
    with pytest.raises(ValueError, match="Shape of velocity field's vx and vy do not match"):
        asda.generate_velocity_field(np.zeros((1, 2)), np.zeros((2, 1)), 0, 0)
    with pytest.raises(TypeError, match="Keyword 'r' must be an integer"):
        asda.generate_velocity_field(np.zeros((1, 2)), np.zeros((1, 2)), 0, 0, 0.8)
    with pytest.raises(ValueError, match="Shape of velocity field's vx and vy do not match"):
        asda.calculate_gamma_values(np.zeros((1, 2)), np.zeros((2, 1)), 0, 0)
    with pytest.raises(TypeError, match="Keyword 'r' must be an integer"):
        asda.calculate_gamma_values(np.zeros((1, 2)), np.zeros((1, 2)), 1, 0.8)
    with pytest.raises(TypeError, match="Keyword 'factor' must be an integer"):
        asda.calculate_gamma_values(np.zeros((1, 2)), np.zeros((1, 2)), 0.8, 3)
    with pytest.raises(TypeError, match="Keyword 'factor' must be an integer"):
        asda.get_vortex_edges(gamma=np.zeros((1, 2)), factor=0.8)
    with pytest.raises(ValueError, match="Shape of velocity field's vx and vy do not match"):
        asda.get_vortex_properties(np.zeros((1, 2)), np.zeros((2, 1)), 0)

    # Generate vx and vy
    with pytest.warns(UserWarning, match="One of the input parameters is missing, setting both to 'None'"):
        vx, vy = asda.get_velocity_field(
            gamma=gamma,
            rcore=rcore,
            ratio_vradial=ratio,
            x_range=[-100, 100, 200],
            y_range=[-100, 100, 200],
            x=np.meshgrid,
        )

    vx, vy = asda.get_velocity_field(
        gamma=gamma,
        rcore=rcore,
        ratio_vradial=ratio,
        x_range=[-100, 100, 200],
        y_range=[-100, 100, 200],
    )

    # perform vortex detection
    gamma = asda.calculate_gamma_values(vx, vy)
    # properties of the detected vortex
    center_edge = asda.get_vortex_edges(gamma)
    (ve, vr, vc, ia) = asda.get_vortex_properties(vx, vy, center_edge)
    np.testing.assert_almost_equal(ve[0], 0.39996991917753405)
    np.testing.assert_almost_equal(vr[0], 1.999849595887626)
    assert vc == ([0.0, 0.0],)
    assert ia == (None,)
    assert len(center_edge) == 5
    np.testing.assert_allclose(center_edge["center"], np.array([[100.0, 100.0]]))
    np.testing.assert_almost_equal(center_edge["peak"], 0.9605688248523583)
    np.testing.assert_almost_equal(center_edge["radius"], 50.0732161286822)
    assert len(center_edge["points"][0]) == 7877
    assert len(center_edge["edge"][0]) == 280

    np.testing.assert_allclose(center_edge["center"][0][0], 100)
    np.testing.assert_allclose(center_edge["center"][0][1], 100)

    np.testing.assert_allclose(vmax, vr[0], atol=0.001)
    np.testing.assert_allclose(vmax * ratio, ve[0], atol=0.001)

    np.testing.assert_allclose(vc[0][0], 0.0)
    np.testing.assert_allclose(vc[0][1], 0.0)
    np.testing.assert_allclose(rmax, center_edge["radius"][0], atol=0.1)


def test_real_data():
    """
    Run the test on real data and compare with the correct answer.

    Notes:
        Input velocity field and image (if there is any) are all stored in
        default Python order (i.e. [y, x] of the data).

        Output gamma values are in the same order, thus the same shape as
        velocity field.

        other outputs are in the order of [x, y], i.e., vc = [vx, vy],
        edge = [[x1, y1], [x2, y2],...], points = [[x1, y1], [x2, y2],...]
        in units of pixel
    """

    # file which stores the velocity field data
    vel_file = get_test_filepath("asda_vxvy.npz")
    # file that stores the correct detection result
    cor_file = get_test_filepath("asda_correct.npz")
    # load velocity field and data
    vxvy = np.load(vel_file, allow_pickle=True)
    vx = vxvy["vx"]
    vy = vxvy["vy"]
    data = vxvy["data"]

    # Perform swirl detection
    factor = 1
    r = 3
    # Gamma1 and Gamma2
    gamma = asda.calculate_gamma_values(vx, vy, factor, r)
    # Determine Swirls
    center_edge = asda.get_vortex_edges(gamma)
    # Properties of Swirls
    ve, vr, vc, ia = asda.get_vortex_properties(vx, vy, center_edge, data)
    # load correct detect results
    correct = dict(np.load(cor_file, allow_pickle=True))

    # compare between detection result and correct detection result
    # number of swirls
    n = len(ve)
    nc = len(correct["ve"])
    assert n == nc

    # find correspondences
    pos = []
    i = 0
    for cen in center_edge["center"]:
        cen = [int(cen[0]), int(cen[1])]
        idx = np.where(correct["center"] == cen)
        assert np.size(idx[0]) >= 2
        pos.append(np.bincount(idx[0]).argmax())

    # perform comparison
    peak_diff = []
    radius_diff = []
    vr_diff = []
    ve_diff = []
    vc_diff = []
    ia_diff = []
    for i in np.arange(n):
        idx = pos[i]
        peak_diff.append((center_edge["peak"][i] - correct["peak"][idx]) / correct["peak"][idx] * 100)
        radius_diff.append((center_edge["radius"][i] - correct["radius"][idx]) / correct["radius"][idx] * 100)
        vr_diff.append((vr[i] - correct["vr"][idx]) / correct["vr"][idx] * 100)
        ve_diff.append((ve[i] - correct["ve"][idx]) / correct["ve"][idx] * 100)
        vc_diff.append((vc[i] - correct["vc"][idx]) / correct["vc"][idx] * 100)
        ia_diff.append((ia[i] - correct["ia"][idx]) / correct["ia"][idx] * 100)

    # Should be no differences
    for diff in [peak_diff, radius_diff, vr_diff, ve_diff, vc_diff, ia_diff]:
        assert np.allclose(np.mean(diff), 0, atol=1e-5)
