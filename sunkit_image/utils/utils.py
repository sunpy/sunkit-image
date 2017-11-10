from __future__ import print_function, division

import numpy as np

import astropy.units as u


def find_pixel_radii(m, scale=None):
    """
    Find the distance of every pixel in a map from the center of the Sun.
    The answer is returned in units of solar radii.

    Parameters
    ----------
    m :
        A sunpy map object.

    scale :


    Returns
    -------
    An array of

    """
    x, y = np.meshgrid(*[np.arange(v.value) for v in m.dimensions]) * u.pix
    hpc_coords = m.pixel_to_world(x, y)
    radii = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2)
    if scale is None:
        return u.R_sun * (radii / m.rsun_obs)
    else:
        return u.R_sun * (radii / scale)

