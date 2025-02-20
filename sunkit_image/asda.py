"""
This module contains an implementation of the Automated Swirl Detection
Algorithm (ASDA).
"""

import warnings
from itertools import product

import numpy as np
from skimage import measure

from sunkit_image.utils import calculate_gamma, points_in_poly, reform2d, remove_duplicate

__all__ = [
    "calculate_gamma_values",
    "generate_velocity_field",
    "get_radial_velocity",
    "get_rotational_velocity",
    "get_velocity_field",
    "get_vortex_edges",
    "get_vortex_meshgrid",
    "get_vortex_properties",
]


def generate_velocity_field(vx, vy, i, j, r=3):
    """
    Given a point ``[i, j]``, generate a velocity field which contains a region
    with a size of ``(2r+1) x (2r+1)`` centered at ``[i, j]`` from the original
    velocity field ``vx`` and ``vy``.

    Parameters
    ----------
    vx : `numpy.ndarray`
        Velocity field in the x direction.
    vy : `numpy.ndarray`
        Velocity field in the y direction.
    i : `int`
        first dimension of the pixel position of a target point.
    j : `int`
        second dimension of the pixel position of a target point.
    r : `int`, optional
        Maximum distance of neighbor points from target point.
        Default value is 3.

    Returns
    -------
    `numpy.ndarray`
        The first dimension is a velocity field which contains a
        region with a size of ``(2r+1) x (2r+1)`` centered at ``[i, j]`` from
        the original velocity field ``vx`` and ``vy``.
        the second dimension is similar as the first dimension, but
        with the mean velocity field subtracted from the original
        velocity field.
    """
    if vx.shape != vy.shape:
        msg = "Shape of velocity field's vx and vy do not match"
        raise ValueError(msg)
    if not isinstance(r, int):
        msg = "Keyword 'r' must be an integer"
        raise TypeError(msg)
    vel = np.array(
        [[vx[i + im, j + jm], vy[i + im, j + jm]] for im in np.arange(-r, r + 1) for jm in np.arange(-r, r + 1)],
    )
    return np.array([vel, vel - vel.mean(axis=0)])


def calculate_gamma_values(vx, vy, factor=1, r=3):
    """
    Calculate ``gamma1`` and ``gamma2`` values of velocity field vx and vy.

    Parameters
    ----------
    vx : `numpy.ndarray`
        Velocity field in the x direction.
    vy : `numpy.ndarray`
        Velocity field in the y direction.
    factor : `int`, optional
        Magnify the original data to find sub-grid vortex center and boundary.
        Default value is 1.
    r : `int`, optional
        Maximum distance of neighbor points from target point.
        Default value is 3.

    Returns
    -------
    `tuple`
        A tuple in form of ``(gamma1, gamma2)``, where ``gamma1`` is useful in
        finding vortex centers and ``gamma2`` is useful in finding vortex
        edges.
    """

    if vx.shape != vy.shape:
        msg = "Shape of velocity field's vx and vy do not match"
        raise ValueError(msg)
    if not isinstance(r, int):
        msg = "Keyword 'r' must be an integer"
        raise TypeError(msg)
    if not isinstance(factor, int):
        msg = "Keyword 'factor' must be an integer"
        raise TypeError(msg)

    # This part of the code was written in (x, y) order
    # but numpy is in (y, x) order so we need to transpose it
    dshape = np.shape(vx)
    vx = vx.T
    vy = vy.T
    if factor > 1:
        vx = reform2d(vx, factor)
        vy = reform2d(vy, factor)
    gamma = np.array([np.zeros_like(vx), np.zeros_like(vy)]).T
    # pm vectors, see equation (8) in Graftieaux et al. 2001 or Equation (1) in Liu et al. 2019
    pm = np.array(
        [[i, j] for i in np.arange(-r, r + 1) for j in np.arange(-r, r + 1)],
        dtype=float,
    )
    # Mode of vector pm
    pnorm = np.linalg.norm(pm, axis=1)
    # Number of points in the concerned region
    N = (2 * r + 1) ** 2

    index = np.array(
        [[i, j] for i in np.arange(r, dshape[0] - r) for j in np.arange(r, dshape[1] - r)],
    )
    index = index.T
    vel = generate_velocity_field(vx, vy, index[1], index[0], r)
    for d, (i, j) in enumerate(
        product(np.arange(r, dshape[0] - r, 1), np.arange(r, dshape[1] - r, 1)),
    ):
        gamma[i, j, 0], gamma[i, j, 1] = calculate_gamma(pm, vel[..., d], pnorm, N)
    # Transpose back vx & vy
    vx = vx.T
    vy = vy.T
    return gamma


def get_vortex_edges(gamma, rmin=4, gamma_min=0.89, factor=1):
    """
    Find all swirls from ``gamma1``, and ``gamma2``.

    Parameters
    ----------
    gamma : `tuple`
        A tuple in form of ``(gamma1, gamma2)``, where ``gamma1`` is useful in
        finding vortex centers and ``gamma2`` is useful in finding vortex
        edges.
    rmin : `int`, optional
        Minimum radius of swirls, all swirls with radius less than ``rmin`` will be rejected.
        Defaults to 4.
    gamma_min : `float`, optional
        Minimum value of ``gamma1``, all potential swirls with
        peak ``gamma1`` values less than ``gamma_min`` will be rejected.
    factor : `int`, optional
        Magnify the original data to find sub-grid vortex center and boundary.
        Default value is 1.

    Returns
    -------
    `dict`
        The keys and their meanings of the dictionary are:
        ``center`` : Center locations of vortices, in the form of ``[x, y]``.
        ``edge`` : Edge locations of vortices, in the form of ``[x, y]``.
        ``points`` : All points within vortices, in the form of ``[x, y]``.
        ``peak`` : Maximum/minimum gamma1 values in vortices.
        ``radius`` : Equivalent radius of vortices.
        All results are in pixel coordinates.
    """
    if not isinstance(factor, int):
        msg = "Keyword 'factor' must be an integer"
        raise TypeError(msg)

    edge_prop = {"center": (), "edge": (), "points": (), "peak": (), "radius": ()}
    cs = np.array(measure.find_contours(gamma[..., 1].T, -2 / np.pi), dtype=object)
    cs_pos = np.array(measure.find_contours(gamma[..., 1].T, 2 / np.pi), dtype=object)
    if len(cs) == 0:
        cs = cs_pos
    elif len(cs_pos) != 0:
        cs = np.append(cs, cs_pos, 0)
    for i in range(np.shape(cs)[0]):
        v = np.rint(cs[i].astype(np.float32))
        v = remove_duplicate(v)
        # Find all points in the contour
        ps = points_in_poly(v)
        dust = [gamma[..., 0][int(p[1]), int(p[0])] for p in ps]
        # Determine swirl properties
        if len(dust) > 1:
            # Effective radius
            re = np.sqrt(np.array(ps).shape[0] / np.pi) / factor
            # Only consider swirls with re >= rmin and maximum gamma1 value greater than gamma_min
            if np.max(np.fabs(dust)) >= gamma_min and re >= rmin:
                # Extract the index, only first dimension
                idx = np.where(np.fabs(dust) == np.max(np.fabs(dust)))[0][0]
                edge_prop["center"] += (np.array(ps[idx]) / factor,)
                edge_prop["edge"] += (np.array(v) / factor,)
                edge_prop["points"] += (np.array(ps) / factor,)
                edge_prop["peak"] += (dust[idx],)
                edge_prop["radius"] += (re,)
    return edge_prop


def get_vortex_properties(vx, vy, edge_prop, image=None):
    """
    Calculate expanding, rotational speed, equivalent radius and average
    intensity of given swirls.

    Parameters
    ----------
    vx : `numpy.ndarray`
        Velocity field in the x direction.
    vy : `numpy.ndarray`
        Velocity field in the y direction.
    edge_prop : `dict`
        The keys and their meanings of the dictionary are:
        ``center`` : Center locations of vortices, in the form of ``[x, y]``.
        ``edge`` : Edge locations of vortices, in the form of ``[x, y]``.
        ``points`` : All points within vortices, in the form of ``[x, y]``.
        ``peak`` : Maximum/minimum gamma1 values in vortices.
        ``radius`` : Equivalent radius of vortices.
        All results are in pixel coordinates.
    image : `numpy.ndarray`
        Has to have the same shape as ``vx`` observational image,
        which will be used to calculate the average observational values of all swirls.

    Returns
    -------
    `tuple`
        The returned tuple has four components, which are:

        ``ve`` : expanding speed, in the same unit as ``vx`` or ``vy``.
        ``vr`` : rotational speed, in the same unit as ``vx`` or ``vy``.
        ``vc`` : velocity of the center, in the form of ``[vx, vy]``.
        ``ia`` : average of the observational values within the vortices if the parameter image is given.
    """
    if vx.shape != vy.shape:
        msg = "Shape of velocity field's vx and vy do not match"
        raise ValueError(msg)

    ve, vr, vc, ia = (), (), (), ()
    for i in range(len(edge_prop["center"])):
        # Centre and edge of i-th swirl
        cen = edge_prop["center"][i]
        edg = edge_prop["edge"][i]
        # Points of i-th swirl
        pnt = np.array(edge_prop["points"][i], dtype=int)
        # Calculate velocity of the center
        vc += (
            [
                vx[int(round(cen[1])), int(round(cen[0]))],
                vy[int(round(cen[1])), int(round(cen[0]))],
            ],
        )
        # Calculate average the observational values
        if image is None:
            ia += (None,)
        else:
            value = sum(image[pos[1], pos[0]] for pos in pnt)
            ia += (value / pnt.shape[0],)
        ve0, vr0 = [], []
        for j in range(edg.shape[0]):
            # Edge position
            idx = [edg[j][0], edg[j][1]]
            # Eadial vector from swirl center to a point at its edge
            pm = [idx[0] - cen[0], idx[1] - cen[1]]
            # Tangential vector
            tn = [cen[1] - idx[1], idx[0] - cen[0]]
            # Velocity vector
            v = [vx[int(idx[1]), int(idx[0])], vy[int(idx[1]), int(idx[0])]]
            ve0.append(np.dot(v, pm) / np.linalg.norm(pm))
            vr0.append(np.dot(v, tn) / np.linalg.norm(tn))
        ve += (np.nanmean(ve0),)
        vr += (np.nanmean(vr0),)
    return ve, vr, vc, ia


def get_vortex_meshgrid(x_range, y_range):
    """
    Returns a meshgrid of the coordinates of the vortex.

    Parameters
    ----------
    x_range : `list`
        Range of the x coordinates of the meshgrid.
    y_range : `list`
        Range of the y coordinates of the meshgrid.

    Return
    ------
    `tuple`
        Contains the meshgrids generated.
    """
    xx, yy = np.meshgrid(np.arange(x_range[0], x_range[1]), np.arange(y_range[0], y_range[1]))
    return xx, yy


def get_rotational_velocity(gamma, rcore, r=0):
    """
    Calculate rotation speed at radius of ``r``.

    Parameters
    ----------
    gamma : `float`, optional
        A replacement for ``vmax`` and only used if both ``gamma`` and ``rcore`` are not `None`.
        Defaults to `None`.
    rcore : `float`, optional
        A replacement for ``rmax`` and only used if both ``gamma`` and ``rcore`` are not `None`.
        Defaults to `None`.
    r : `float`, optional
        Radius which defaults to 0.

    Return
    ------
    `float`
        Rotating speed at radius of ``r``.
    """
    r = r + 1e-10
    return gamma * (1.0 - np.exp(0 - np.square(r) / np.square(rcore))) / (2 * np.pi * r)


def get_radial_velocity(gamma, rcore, ratio_vradial, r=0):
    """
    Calculate radial (expanding or shrinking) speed at radius of ``r``.

    Parameters
    ----------
    gamma : `float`, optional
        A replacement for ``vmax`` and only used if both ``gamma`` and ``rcore`` are not `None`.
        Defaults to `None`.
    rcore : `float`, optional
        A replacement for ``rmax`` and only used if both ``gamma`` and ``rcore`` are not `None`.
        Defaults to `None`.
    ratio_vradial : `float`, optional
        Ratio between expanding/shrinking speed and rotating speed.
        Defaults to 0.
    r : `float`, optional
        Radius which defaults to 0.

    Return
    ------
    `float`
        Radial speed at the radius of ``r``.
    """
    r = r + 1e-10
    return get_rotational_velocity(gamma, rcore, r) * ratio_vradial


def get_velocity_field(gamma, rcore, ratio_vradial, x_range, y_range, x=None, y=None):
    """
    Calculates the velocity field in a meshgrid generated with ``x_range`` and
    ``y_range``.

    Parameters
    ----------
    gamma : `float`, optional
        A replacement for ``vmax`` and only used if both ``gamma`` and ``rcore`` are not `None`.
        Defaults to `None`.
    rcore : `float`, optional
        A replacement for ``rmax`` and only used if both ``gamma`` and ``rcore`` are not `None`.
        Defaults to `None`.
    ratio_vradial : `float`, optional
        Ratio between expanding/shrinking speed and rotating speed.
        Defaults to 0.
    x_range : `list`
        Range of the x coordinates of the meshgrid.
    y_range : `list`
        range of the y coordinates of the meshgrid.
    x, y : `numpy.meshgrid`, optional
        If both are given, ``x_range`` and ``y_range`` will be ignored.
        Defaults to None``.

    Return
    ------
    `tuple`
        The generated velocity field ``(vx, vy)``.
    """
    if x is None or y is None:
        # Check if one of the input parameters is None but the other one is not None
        if x != y:
            warnings.warn("One of the input parameters is missing, setting both to 'None'", stacklevel=3)
            x, y = None, None
        # Creating mesh grid
        x, y = get_vortex_meshgrid(x_range=x_range, y_range=y_range)
    # Calculate radius
    r = np.sqrt(np.square(x) + np.square(y)) + 1e-10
    # Calculate velocity vector
    vector = [
        0 - get_rotational_velocity(gamma, rcore, r) * y + get_radial_velocity(gamma, rcore, ratio_vradial, r) * x,
        get_rotational_velocity(gamma, rcore, r) * x + get_radial_velocity(gamma, rcore, ratio_vradial, r) * y,
    ]
    vx = vector[0] / r
    vy = vector[1] / r
    return vx, vy
