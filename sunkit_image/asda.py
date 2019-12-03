"""
This module contains the implementation of the Automated Swirl Detection
Algorithm (ASDA). See Graftieaux et al. 2001 and Liu et al. 2019 for details.
"""

import numpy as np
import warnings as wr
import matplotlib.pyplot as plt
from itertools import product
from skimage import measure
from sunkit_image.utils import (
    reform2d,
    points_in_poly,
    remove_duplicate
)


__author__ = ['Jiajia Liu', 'Norbert Gyenge']
__email__ = ['jj.liu@sheffield.ac.uk']


class Asda_Calc:
    """
    Class for ASDA
    """

    def __init__(self, vx, vy, r=3, factor=1):
        """
        Initialise the Asda_Calc class

        Parameters
        ----------
            vx : `list` or `numpy.ndarray`
                velocity field in x direction.
            vy : `list` or `numpy.ndarray`
                velocity field in y direction.
            r : `int`
                maximum distance of neighbour points from target point,
                default value is 3. See definition in Graftieaux et al. 2001
                and Liu et al. 2019.
            factor : `int`
                Magnify the original data to find sub-grid vortex center and
                boundary. This is EXPERIMENTAL. Default value is 1.

        Returns
        -------
        A Python class of Asda_Calc

        References
        ----------
        * Laurent Graftieaux, Marc Michard and Nathalie Grosjean.
          Combining PIV, POD and vortex identification algorithms for the
          study of unsteady turbulent swirling flows.
          Meas. Sci. Technol. 12, 1422, 2001.
        * Jiajia Liu, Chris Nelson, Robert Erdélyi.
          Automated Swirl Detection Algorithm (ASDA) and Its Application to
          Simulation and Observational Data.
          Astrophys. J., 872, 22, 2019.
        """

        # Initialise class properties
        self.vx = np.array(vx, dtype=np.float32)
        self.vy = np.array(vy, dtype=np.float32)
        self.dshape = np.shape(vx)
        self.r = r
        self.factor = factor

        # Check the dimensions of the velocity fields
        if self.vx.shape != self.vy.shape:
            raise ValueError("Velocity field vx and vy do not match!")

        # Check input parameters
        if not isinstance(self.r, int):
            raise ValueError("Parameter 'r' must be an integer!")

        if not isinstance(self.factor, int):
            raise ValueError("Parameter 'factor' must be an integer!")

    def gamma_values(self):
        """
        Calculate gamma1 and gamma2 values of velocity field vx and vy
        Fomulars can be found in Graftieaux et al. 2001 & Liu et al. 2019.

        Parameters
        ----------
        None

        Returns
        -------
            `tuple`
            a tuple in form of (gamma1, gamma2), where gamma1 is useful in
            finding vortex centers and gamma2 is useful in finding vortex
            edges.
        """

        def gen_vel(i, j):
            """
            Given a point [i, j], generate a velocity field which contains
            a region with a size of (2r+1) x (2r+1) centered at [i, j] from
            the original velocity field self.vx and self.vy.

            Parameters
            ----------
            i : `int`
                first dimension of the pixel position of a target point.
            j : `int`
                second dimension of the pixel position of a target point.

            Returns:
            -------
                `numpy.ndarray`
                the first dimension is a velocity field which contains a
                region with a size of (2r+1) x (2r+1) centered at [i, j] from
                the original velocity field self.vx and self.vy.
                the second dimension is similar as the first dimension, but
                with the mean velocity field substracted from the original
                velocity field.
            """

            vel = np.array([[self.vx[i + im, j + jm], self.vy[i + im, j + jm]]
                            for im in np.arange(-self.r, self.r + 1)
                            for jm in np.arange(-self.r, self.r + 1)])

            return np.array([vel, vel - vel.mean(axis=0)])

        def calc_gamma(pm, vel, pnorm, N):
            """
            Calculate Gamma values, see equation (8) in Graftieaux et al. 2001
            or Equation (1) in Liu et al. 2019

            Parameters
            ----------
                pm : `numpy.ndarray`
                    vector from point p to point m
                vel : `numpy.ndarray`
                    velocity vector
                pnorm : `numpy.ndarray`
                    mode of pm
                N : `int`
                    number of points

            Returns
            -------
                `numpy.ndarray`
                calculated gamma values for velocity vector vel
            """

            cross = np.cross(pm, vel)
            vel_norm = np.linalg.norm(vel, axis=2)
            sint = cross / (pnorm * vel_norm + 1e-10)

            return np.nansum(sint, axis=1) / N

        # this part of the code was written in (x, y) order
        # but default Python is in (y, x) order
        # so we need to transpose it
        self.vx = self.vx.T
        self.vy = self.vy.T

        # reform data is factor is greater than 1
        if self.factor > 1:
            self.vx = reform2d(self.vx, self.factor)
            self.vy = reform2d(self.vy, self.factor)

        # Initialise Gamma1 and Gamma2
        self.gamma = np.array([np.zeros_like(self.vx),
                               np.zeros_like(self.vy)]).T

        # pm vectors, see equation (8) in Graftieaux et al. 2001 or Equation
        # (1) in Liu et al. 2019
        pm = np.array([[i, j]
                       for i in np.arange(-self.r, self.r + 1)
                       for j in np.arange(-self.r, self.r + 1)], dtype=float)

        # mode of vector pm
        pnorm = np.linalg.norm(pm, axis=1)

        # Number of points in the concerned region
        N = (2 * self.r + 1) ** 2

        # Create index array
        index = np.array([[i, j]
                         for i in np.arange(self.r, self.dshape[0] - self.r)
                         for j in np.arange(self.r, self.dshape[1] - self.r)])

        # Transpose index
        index = index.T

        # Generate velocity field
        vel = gen_vel(index[1], index[0])

        # Iterate over the array gamma
        for d, (i, j) in enumerate(product(np.arange(self.r,
                                                     self.dshape[0] - self.r,
                                                     1),
                                           np.arange(self.r,
                                                     self.dshape[1] - self.r,
                                                     1))):

            self.gamma[i, j, 0], self.gamma[i, j, 1] = calc_gamma(pm,
                                                                  vel[..., d],
                                                                  pnorm, N)

        # Transpose back vx & vy
        self.vx = self.vx.T
        self.vy = self.vy.T

        return self.gamma

    def center_edge(self, rmin=4, gamma_min=0.89):
        """
        Find all swirls from gamma1, and gamma2

        Parameters
        ----------
            rmin : `int`
                minimum radius of swirls, all swirls with radius less than
                rmin will be rejected.
            gamma_min : `float`
                minimum value of gamma1, all potential swirls with
                peak gamma1 values less than gamma_min will be rejected.

        Returns
        -------
            `dictionary`
            The keys and their meanings of the dictionary are:
            center: center locations of vortices, in the form of [x, y].
            edge: edge locations of vortices, in the form of [x, y].
            points: all points within vortices, in the form of [x, y].
            peak: maximum/minimum gamma1 values in vortices.
            radius: equivalent radius of vortices.
            All results are in pixel coordinates.
        """

        # Initial dictionary setup
        self.edge_prop = {'center': (), 'edge': (), 'points': (), 'peak': (),
                          'radius': ()}
        # ------------ Old algorithm, depracated -----------------------------
        # # Turn interactive plotting off
        # plt.ioff()
        # plt.figure(-1)
        # # Find countours
        # cs = plt.contour(self.gamma[..., 1], levels=[-2 / np.pi, 2 / np.pi])
        # plt.close(-1)

        # # iterate over all contours
        # for i in range(len(cs.collections)):
        #     # Extract a contour and iterate over
        #     for c in cs.collections[i].get_paths():
        #         # convert the single contour to list
        #         v = c.vertices
        #         v = np.rint(c.vertices).tolist()
        # ------------ Old algorithm, depracated -----------------------------

        cs = np.array(measure.find_contours(self.gamma[..., 1].T, -2 / np.pi))
        cs_pos = np.array(measure.find_contours(self.gamma[..., 1].T,
                                                2 / np.pi))
        if len(cs) == 0:
            cs = cs_pos
        elif len(cs_pos) != 0:
            cs = np.append(cs, cs_pos, 0)
        for i in range(np.shape(cs)[0]):
            v = np.rint(cs[i])
            v = remove_duplicate(v)
            # find all points in the contour
            ps = points_in_poly(v)
            # gamma1 value of all points in the contour
            dust = []
            for p in ps:
                dust.append(self.gamma[..., 0][int(p[1]), int(p[0])])

            # determin swirl properties
            if len(dust) > 1:
                # effective radius
                re = np.sqrt(np.array(ps).shape[0] / np.pi) / self.factor
                # only consider swirls with re >= rmin and maximum gamma1
                # value greater than gamma_min
                if np.max(np.fabs(dust)) >= gamma_min and re >= rmin:
                    # Extract the index, only first dimension
                    idx = np.where(np.fabs(dust) ==
                                   np.max(np.fabs(dust)))[0][0]
                    # Update dictionary key 'center'
                    self.edge_prop['center'] += \
                        (np.array(ps[idx]) / self.factor, )
                    # Update dictionary key 'edge'
                    self.edge_prop['edge'] += \
                        (np.array(v) / self.factor, )
                    # Update dictionary key 'points'
                    self.edge_prop['points'] += \
                        (np.array(ps) / self.factor, )
                    # Update dictionary key 'peak'
                    self.edge_prop['peak'] += (dust[idx],)
                    # Update dictionary key 'radius'
                    self.edge_prop['radius'] += (re,)

        return self.edge_prop

    def vortex_property(self, image=None):
        """
        Calculate expanding, rotational speed, equivalent radius and average
        intensity of given swirls.

        Parameters
        ----------
            image : `list` or `numpy.ndarray` with the same shape as self.vx
                observational image, which will be used to calculate the
                average observational values of all swirls.

        Outputs
        -------
            `tuple`
            The returned tuple has four components, each component in order is
            ve: expanding speed, in the same unit as self.vx or self.vy.
            vr: rotational speed, in the same unit as self.vx or self.vy.
            vc: velocity of the center, in the form of [vx, vy].
            ia: average of the observational values (intensity or magnetic
                field strength etc.) within the vortices if the parameter
                image is given.
        """

        # Initialising containers
        ve, vr, vc, ia = (), (), (), ()

        # Iterate over the swirls
        for i in range(len(self.edge_prop['center'])):

            # Centre and edge of i-th swirl
            cen = self.edge_prop['center'][i]
            edg = self.edge_prop['edge'][i]

            # Points of i-th swirl
            pnt = np.array(self.edge_prop['points'][i], dtype=int)

            # Calculate velocity of the center
            vc += ([self.vx[int(round(cen[1])), int(round(cen[0]))],
                    self.vy[int(round(cen[1])), int(round(cen[0]))]],)

            # Calculate average the observational values
            if image is None:

                # Appening 'ia' with None if no image
                ia += (None, )
            else:

                # Calculate ia
                value = 0
                for pos in pnt:
                    value += image[pos[1], pos[0]]

                # Appending 'ia'
                ia += (value / pnt.shape[0], )

            # Clearing list ve0 and vr0
            ve0, vr0 = [], []

            # Iterate over the shapes
            for j in range(edg.shape[0]):

                # Edge position
                idx = [edg[j][0], edg[j][1]]

                # radial vector from swirl center to a point at its edge
                pm = [idx[0] - cen[0], idx[1] - cen[1]]

                # tangential vector
                tn = [cen[1] - idx[1], idx[0] - cen[0]]

                # velocity vector
                v = [self.vx[int(idx[1]), int(idx[0])],
                     self.vy[int(idx[1]), int(idx[0])]]

                # Appending ve0 amd vr0
                ve0.append(np.dot(v, pm) / np.linalg.norm(pm))
                vr0.append(np.dot(v, tn) / np.linalg.norm(tn))

            # Appending ve and vt
            ve += (np.nanmean(ve0),)
            vr += (np.nanmean(vr0),)

        return (ve, vr, vc, ia)

    def visual_gamma(self, gamma2=False, fname=None, origin='lower', **kwargs):
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
            title = r'$\Gamma_2$'
        else:
            # Select gamma1
            gamma = self.gamma[..., 0]
            # Plot title:
            title = r'$\Gamma_1$'

        # creat the plot
        fig, ax = plt.subplots(figsize=(6, 6.0 * self.dshape[0] /
                                        self.dshape[1]))
        fig.canvas.set_window_title('Gamma Value')

        # Show the image
        ax.imshow(gamma, origin=origin, **kwargs)

        # Set image title
        ax.set_title(title)

        # Set axis labesl
        ax.set(xlabel='x', ylabel='y')

        if fname is None:
            plt.show()
        else:
            plt.savefig(fname, **kwargs)


class Lamb_Oseen(Asda_Calc):

    """
    Creating an artifactual Lamb Oseen vortex

    Reference
    ---------
    https://en.wikipedia.org/wiki/Lamb%E2%80%93Oseen_vortex
    """

    def __init__(self, vmax=2.0, rmax=5, gamma=None, rcore=None,
                 ratio_vradial=0, factor=1, r=3):
        """
        Initialization of the Lamb Oseen vortex

        Parameters
        ----------
        vmax : `float`
            rotating speed of the vortex, negative value for clockwise vortex.
        rmax : `float`
            radius of of the vortex.
        ratio_vradial : `float`
            ratio between expanding/shrinking speed and rotating speed.
        gamma : `float`
        rcore : `float`
            gamma and rcore are replacements for vamx and rmax. If both
            are not None, will use gamma and rcore instread of vmax and rmax.
            See definition for Lamb Oseen vortex.
        
        Returns
        -------
            None

        """

        # alpha of Lamb Oseen vortices
        self.alpha = 1.256430
        self.ratio_vradial = ratio_vradial

        if gamma is None or rcore is None:

            # Check if one of the input parameters is None but the other one
            # is not None
            if (gamma is None) != (rcore is None):

                # Missing input parameter
                wr.warn("One of the input parameters is missing," +
                        "setting both to 'None'")
                gamma, rcore = None, None

            # Radius of the position where v_theta reaches vmax
            self.rmax = rmax

            # Maximum value of v_theta
            self.vmax = vmax

            # Core radius
            self.rcore = self.rmax / np.sqrt(self.alpha)
            self.gamma = 2 * np.pi * self.vmax * self.rmax * \
                (1 + 1 / (2 * self.alpha))
        else:

            # radius
            self.rmax = self.rcore * np.sqrt(self.alpha)

            # rotating speed
            self.vmax = self.gamma / (2 * np.pi * self.rmax *
                                      (1 + 1 / (2 * self.alpha)))

            # core radius
            self.rcore = rcore
            self.gamma = gamma

        # Calculating core speed
        self.vcore = (1 - np.exp(-1.0)) * self.gamma / (2 * np.pi * self.rcore)
        self.r = r
        self.factor = factor

        # check input parameter r and factor
        if isinstance(self.r, int) is False:
            raise Exception("Parameter 'r' must be an integer!")

        if isinstance(self.factor, int) is False:
            raise Exception("Parameter 'factor' must be an integer!")

    def get_grid(self, x_range, y_range):
        """
        Return meshgrid of the coordinate of the vortex

        Parameters
        ----------
        x_range : `list`
            range of the x coordinates of the meshgrid.
        y_range : `list`
            range of the y coordinates of the meshgrid.

        Return
        ------
            `tuple`
            contains the meshgrids generated.
        """

        self.xx, self.yy = np.meshgrid(np.arange(x_range[0], x_range[1]),
                                       np.arange(y_range[0], y_range[1]))
        self.dshape = np.shape(self.xx)

        return self.xx, self.yy

    def get_vtheta(self, r=0):
        """
        Calculate rotating speed at radius of r.

        Parameters
        ----------
        r : `float`
            radius.

        Return
        ------
            `float`
            rotating speed at radius of r.
        """

        r = r + 1e-10
        return self.gamma * (1.0 - np.exp(0 - np.square(r) /
                                          np.square(self.rcore))) / \
            (2 * np.pi * r)

    def get_vradial(self, r=0):
        """
        Calculate radial (expanding or shrinking) speed at radius of r.

        Parameters
        ----------
        r : `float`
            radius.

        Return
        ------
            `float`
            radial speed at the radius of r.
        """

        r = r + 1e-10
        return self.get_vtheta(r) * self.ratio_vradial

    def get_vxvy(self, x_range, y_range, x=None, y=None):
        """
        calculate velocity field in a meshgrid generated with x_range and
        y_range

        Parameters
        ----------
        x_range : `list`
            range of the x coordinates of the meshgrid.
        y_range : `list`
            range of the y coordinates of the meshgrid.
        x, y : `numpy.meshgrid`
            If both are given, x_range and y_range will be ignored.

        Return
        ------
        generated velocity field
        """

        # Check the dimensions of x_range
        if len(x_range) != 2:
            self.x_range = [0 - self.rmax, self.rmax]

        # Check the dimensions of y_range
        if len(y_range) != 2:
            self.y_range = [0 - self.rmax, self.rmax]

        if (x is None) or (y is None):

            # Check if one of the input parameters is None
            # but the other one is not None
            if (x is None) != (y is None):
                wr.warn("One of the input parameters is missing, setting " +
                        " both to 'None'")
                x, y = None, None

            # Creating mesh grid
            x, y = self.get_grid(x_range=x_range, y_range=y_range)

        # calculate radius
        r = np.sqrt(np.square(x) + np.square(y)) + 1e-10

        # calculate velocity vector
        vector = [0 - self.get_vtheta(r) * y + self.get_vradial(r) * x,
                  self.get_vtheta(r) * x + self.get_vradial(r) * y]

        self.vx = vector[0] / r
        self.vy = vector[1] / r

        return self.vx, self.vy

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
        fig, ax = plt.subplots(figsize=(6, 6.0 * self.dshape[0] /
                                        self.dshape[1]))

        # set window title
        fig.canvas.set_window_title('Lamb-Oseen Vortex')

        # Set image title
        ax.set_title('Lamb-Oseen Vortex')

        # Generate a stream plot
        ax.streamplot(self.xx, self.yy, self.vx, self.vy, **kwargs)

        # Set axis labesl
        ax.set(xlabel='x', ylabel='y')

        # save file if fname is not None
        if fname is None:
            plt.show()

        else:
            plt.savefig(fname, **kwargs)
