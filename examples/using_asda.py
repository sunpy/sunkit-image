"""
================
Detecting Swirls
================

This example showcases how to use Automated Swirl Detection Algorithm (ASDA) to
detect swirls in velocity fields.
"""

from datetime import datetime

import numpy as np

from sunkit_image.asda import Asda

###########################################################################
# This examples demonstrates find swirls in a 2D velocity flow field.
# We will use precomputed a flow field from our test data.


def asda_example():
    """

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
    vel_file = download_file("https://github.com/PyDL/asda-class/blob/master/data/vxvy.npz")
    # file that stores the correct detection result
    cor_file = download_file("https://github.com/PyDL/asda-class/blob/master/data/correct.npz")
    # load velocity field and data
    vxvy = np.load(vel_file)
    vx = vxvy["vx"]
    vy = vxvy["vy"]
    data = vxvy["data"]

    # Perform swirl detection
    factor = 1
    # Initialise class
    lo = Asda(vx, vy, factor=factor)
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
        peak_diff.append((center_edge["peak"][i] - correct["peak"][idx]) / correct["peak"][idx])
        radius_diff.append((center_edge["radius"][i] - correct["radius"][idx]) / correct["radius"][idx])
        vr_diff.append((vr[i] - correct["vr"][idx]) / correct["vr"][idx])
        ve_diff.append((ve[i] - correct["ve"][idx]) / correct["ve"][idx])
        vc_diff.append((vc[i] - correct["vc"][idx]) / correct["vc"][idx])
        ia_diff.append((ia[i] - correct["ia"][idx]) / correct["ia"][idx])

    print("Difference in Peak Gamma1 Value:", np.max(peak_diff), np.mean(peak_diff), np.min(peak_diff))
    print("Difference in radius:", np.max(radius_diff), np.mean(radius_diff), np.min(radius_diff))
    print("Difference in rotating speed:", np.max(vr_diff), np.mean(vr_diff), np.min(vr_diff))
    print("Difference in expanding speed:", np.max(ve_diff), np.mean(ve_diff), np.min(ve_diff))
    print("Difference in average intensity:", np.max(ia_diff), np.mean(ia_diff), np.min(ia_diff))

    def visual_vortex(self, fname=None, **kwargs):
        """
        Visualise the vortex.

        Parameters
        ----------
        fname : `string`
            file to be saved. The default is None, no image will be saved.
        **kwargs :
            keywords for pyplot.imshow and pyplot.savefig.

        Returns
        -------
            None.
        """

        # creat the figure
        fig, ax = plt.subplots(figsize=(6, 6.0 * self.dshape[0] / self.dshape[1]))

        # set window title
        fig.canvas.set_window_title("Lamb-Oseen Vortex")

        # Set image title
        ax.set_title("Lamb-Oseen Vortex")

        # Generate a stream plot
        ax.streamplot(self.xx, self.yy, self.vx, self.vy, **kwargs)

        # Set axis labesl
        ax.set(xlabel="x", ylabel="y")

        # save file if fname is not None
        if fname is None:
            plt.show()

        else:
            plt.savefig(fname, **kwargs)

    def visual_gamma(self, gamma2=False, fname=None, origin="lower", **kwargs):
        """
        Visualise Gamma1 or Gamma2 (if gamma2 is set to True)

        Parameters
        ----------
        gamma2 : `Bool`
            The default is False. If set, will visualise gamma2 instead.
        fname : `string`
            file to be saved. The default is None. If not set, no image will
            be saved.
        origin : `string`
            Origin of the image. The default is 'lower'.
        **kwargs :
            keywords for pyplot.imshow and pyplot.savefig.

        Returns
        -------
        None.
        """
        if gamma2:
            # Select gamma2
            gamma = self.gamma[..., 1]
            # Plot title:
            title = r"$\Gamma_2$"
        else:
            # Select gamma1
            gamma = self.gamma[..., 0]
            # Plot title:
            title = r"$\Gamma_1$"

        # creat the plot
        fig, ax = plt.subplots(figsize=(6, 6.0 * self.dshape[0] / self.dshape[1]))
        fig.canvas.set_window_title("Gamma Value")

        # Show the image
        ax.imshow(gamma, origin=origin, **kwargs)

        # Set image title
        ax.set_title(title)

        # Set axis labesl
        ax.set(xlabel="x", ylabel="y")

        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, **kwargs)
