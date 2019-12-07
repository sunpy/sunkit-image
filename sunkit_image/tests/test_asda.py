from datetime import datetime

import numpy as np
import pytest

import astropy

from sunkit_image import asda


def test_artificial():
    """
    Generate an artificial vortex using the Lamb_Oseen class in asda, then
    perform the vortex detection and visualisation.
    """
    # Generate an artificial vortex
    vmax = 2.0  # rotating speed
    rmax = 50  # radius
    ratio = 0.2  # ratio of expanding speed over rotating speed
    lo = asda.Lamb_Oseen(vmax=vmax, rmax=rmax, ratio_vradial=ratio, factor=1)
    # Generate vx and vy
    vx, vy = lo.get_vxvy(x_range=[-100, 100], y_range=[-100, 100])

    # perform vortex detection
    beg_time = datetime.today()
    lo.gamma_values()
    # time used
    end_time = datetime.today()
    print("Time used for calculating Gamma1 & Gamma2", end_time - beg_time)
    # properties of the detected vortex
    center_edge = lo.center_edge()
    (ve, vr, vc, ia) = lo.vortex_property()
    print("Vortex Parameters:")
    print(
        "Center Location ({:6.2f}, {:6.2f}),".format(100, 100),
        "Detection is ({:6.2f}, {:6.2f})".format(center_edge["center"][0][0], center_edge["center"][0][1]),
    )
    print("Rotating Speed {:6.2f}, Detection is {:6.2f}".format(vmax, vr[0]))
    print("Expanding Speed {:6.2f}, Detection is {:6.2f}".format(vmax * ratio, ve[0]))
    print(
        ("Center Speed ({:6.2f}, {:6.2f}), " + "Detection is ({:6.2f}, {:6.2f})").format(
            0.0, 0.0, vc[0][0], vc[0][1]
        )
    )
    print("Radius {:6.2f}, Detection is {:6.2f}".format(rmax, center_edge["radius"][0]))

    # Visualise vortex
    lo.visual_vortex()
    # visualise gamma2
    lo.visual_gamma(gamma2=True)


@pytest.mark.remote_data
def test_real_data():
    """
    run the test on real data and compare with the correct answer.

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
    vel_file = astropy.utils.data.download_file(
        "https://raw.githubusercontent.com/PyDL/asda-class/master/data/vxvy.npz"
    )
    # file that stores the correct detection result
    cor_file = astropy.utils.data.download_file(
        "https://raw.githubusercontent.com/PyDL/asda-class/master/data/correct.npz"
    )
    # load velocity field and data
    vxvy = np.load(vel_file, allow_pickle=True)
    vx = vxvy["vx"]
    vy = vxvy["vy"]
    data = vxvy["data"]

    # Perform swirl detection
    factor = 1
    # Initialise class
    lo = asda.Asda(vx, vy, factor=factor)
    # Gamma1 and Gamma2
    beg_time = datetime.today()
    lo.gamma_values()
    # Caculate time consumption
    end_time = datetime.today()
    print("Time used for calculating Gamma1 & Gamma2", end_time - beg_time)
    # Determine Swirls
    center_edge = lo.center_edge()
    # Properties of Swirls
    ve, vr, vc, ia = lo.vortex_property(image=data)
    # load correct detect results
    correct = dict(np.load(cor_file, allow_pickle=True))

    # visualise gamma2
    lo.visual_gamma(gamma2=True)

    # compare between detection result and correct detection result
    # number of swirls
    n = len(ve)
    nc = len(correct["ve"])
    if n != nc:
        raise Exception("The number of swirls is wrong!")

    # find correspondances
    pos = []
    i = 0
    for cen in center_edge["center"]:
        cen = [int(cen[0]), int(cen[1])]
        idx = np.where(correct["center"] == cen)
        if np.size(idx[0]) < 2:
            raise Exception("At least one swirl is not in the correct" + " position")
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

    # Print out difference with correct detection result
    print(
        "The relative difference (%) between swirls detected by this "
        + "program and the correct one is as following:"
    )
    print("----------------------")
    print(
        ("Difference in Peak Gamma1 Value: max {:6.2f}%, mean {:6.2f}%," + " min {:6.2f}%").format(
            np.max(peak_diff), np.mean(peak_diff), np.min(peak_diff)
        )
    )
    print(
        ("Difference in radius: max {:6.2f}%, mean {:6.2f}%," + " min {:6.2f}%").format(
            np.max(radius_diff), np.mean(radius_diff), np.min(radius_diff)
        )
    )
    print(
        ("Difference in rotating speed: max {:6.2f}%, mean {:6.2f}%," + " min {:6.2f}%").format(
            np.max(vr_diff), np.mean(vr_diff), np.min(vr_diff)
        )
    )
    print(
        ("Difference in expanding/shriking speed: max {:6.2f}%, " + "mean {:6.2f}%, min {:6.2f}").format(
            np.max(ve_diff), np.mean(ve_diff), np.min(ve_diff)
        )
    )
    print(
        ("Difference in average intensity: max {:6.2f}%, mean {:6.2f}%," + " min {:6.2f}%").format(
            np.max(ia_diff), np.mean(ia_diff), np.min(ia_diff)
        )
    )
