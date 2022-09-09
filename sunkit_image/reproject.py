"""
Functionality for reprojecting solar images between different coordinate
systems.

This sub-module does not contain functions for actually reprojecting maps,
but instead contains helpers for creating new map headers for coordinate
systems that are commonly used in solar physics. To perform the actual
reprojection on a map, the :meth:`~sunpy.map.GenericMap.reproject_to`
method can be used with the generated headers.

For an example, see :ref:`sphx_glr_generated_gallery_reproject_carrington.py`.
"""
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.map import make_fitswcs_header

__all__ = ["carrington_header"]


def carrington_header(smap, *, shape_out, projection_code="CAR"):
    """
    Construct a FITS-WCS header for a Carrington coordinate frame.

    The date-time and observer coordinate of the new coordinate frame
    are taken from the input map. The resulting WCS covers the full surface
    of the Sun, and has a reference coordinate at (0, 0) degrees Carrington
    Longitude/Latitude.

    Parameters
    ----------
    smap : sunpy.map.GenericMap
        Input map.
    shape_out : [int, int]
        Output map shape, number of pixels in (latitude, longitude).
    projection_code : {'CAR', 'CEA'}
        Projection to use for the latitude coordinate.

    Returns
    -------
    sunpy.map.MetaDict
    """
    valid_codes = {"CAR", "CEA"}
    if projection_code not in valid_codes:
        raise ValueError(f"projection_code must be one of {valid_codes}")

    frame_out = SkyCoord(
        0 * u.deg,
        0 * u.deg,
        frame="heliographic_carrington",
        obstime=smap.date,
        observer=smap.observer_coordinate,
    )

    if projection_code == "CAR":
        scale = [180 / int(shape_out[0]), 360 / int(shape_out[1])] * u.deg / u.pix
    elif projection_code == "CEA":
        # Since, this map uses the cylindrical equal-area (CEA) projection,
        # the spacing needs to be to 180/pi times the sin(latitude)
        # spacing
        # Reference: Section 5.5, Thompson 2006
        scale = [180 / int(shape_out[0] / (np.pi * 2)), 360 / int(shape_out[1])] * u.deg / u.pix
    header = make_fitswcs_header(shape_out, frame_out, scale=scale, projection_code=projection_code)
    return header
