import warnings
from typing import NamedTuple

import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.util.exceptions import SunpyUserWarning

from sunkit_image.coalignment.util.decorators import registered_methods

__all__ = ["coalignment", "affine_params"]


class affine_params(NamedTuple):
    """
    A named tuple to store the affine transformation parameters used for
    updating the metadata.

    Attributes
    ----------
    scale : tuple[tuple[float, float], tuple[float, float]]
        The scale matrix representing the scaling transformation.
        This 2x2 matrix defines how the image is scaled along the x and y axes.

    rotation_matrix : tuple[tuple[float, float], tuple[float, float]]
        The rotation matrix representing the rotation transformation.
        This 2x2 matrix defines the rotation of the image in the plane.

    translation : tuple[float, float]
        The translation vector representing the translation transformation.
        This 2-element tuple defines the shift of the image along the x and y axes in pixels.
    """
    scale: tuple[tuple[float, float], tuple[float, float]]
    rotation_matrix: tuple[tuple[float, float], tuple[float, float]]
    translation: tuple[float, float]


def update_fits_wcs_metadata(reference_map, target_map, affine_params):
    """
    Update the metadata of a sunpy Map object based on affine transformation
    parameters.

    Parameters
    ----------
    target_map : `sunpy.map.Map`
        The original map object whose metadata is to be updated.
    affine_params : object
        An object containing the affine transformation parameters. This object must
        have attributes for translation (dx, dy), scale, and rotation.

    Returns
    -------
    `sunpy.map.Map`
        A new sunpy map object with updated metadata reflecting the affine transformation.
    """
    target_map.reference_pixel
    pc_matrix = target_map.rotation_matrix
    # Extacting the affine parameters
    translation = affine_params.translation
    scale = affine_params.scale
    rotation_matrix = affine_params.rotation_matrix
    # Updating the PC matrix
    new_pc_matrix = pc_matrix @ rotation_matrix @ scale

    reference_coord = reference_map.pixel_to_world(translation[0], translation[1])
    Txshift = reference_coord.Tx - target_map.bottom_left_coord.Tx
    Tyshift = reference_coord.Ty - target_map.bottom_left_coord.Ty
    # Create a new map with the updated metadata
    fixed_map = target_map.shift_reference_coord(Txshift, Tyshift)

    fixed_map.meta["PC1_1"] = new_pc_matrix[0, 0]
    fixed_map.meta["PC1_2"] = new_pc_matrix[0, 1]
    fixed_map.meta["PC2_1"] = new_pc_matrix[1, 0]
    fixed_map.meta["PC2_2"] = new_pc_matrix[1, 1]

    return fixed_map

def warn_user_of_nan(array, name):
    """
    Issues a warning if there are NaN values in the input array.

    Parameters
    ----------
    array : `numpy.ndarray`
        The input array to be checked for NaN values.
    name : str
        The name of the array, used in the warning message.
    """
    if not np.all(np.isfinite(array)):
        warnings.warn(
            f"The {name} map has nonfinite entries. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure there are no infinity or "
            "Not a Number values. For instance, replacing them with a "
            "local mean.",
            SunpyUserWarning,
            stacklevel=3,
        )


def warn_user_of_separation(reference_map,target_map):
    """
    Issues a warning if the separation between the reference and target maps is
    large.

    Parameters
    ----------
    reference_map : sunpy.map.Map
        The reference map to which the target map is to be coaligned.
    target_map : sunpy.map.Map
        The target map to be coaligned to the reference map.
    """
    # Calculate separation between the reference and target maps
    ref_coord = SkyCoord(reference_map.observer_coordinate)
    target_coord = SkyCoord(target_map.observer_coordinate)
    angular_separation = ref_coord.separation(target_coord)
    if angular_separation > (1*u.deg):
        warnings.warn(
            "The separation between the reference and target maps is large. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure the maps are close in time and space.",
            SunpyUserWarning,
            stacklevel=3,
        )
    # Calculate time difference and convert to separation angle
    ref_time = reference_map.date
    target_time = target_map.date
    time_diff = abs(ref_time - target_time)
    solar_rotation_rate = 13.33*u.deg / u.day ### Verify this value
    time_separation_angle = (time_diff.to(u.day) * solar_rotation_rate).to(u.deg)
    if time_separation_angle > (1*u.deg):
        warnings.warn(
            "The separation between the reference and target maps in time is large. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure the maps are close in time and space.",
            SunpyUserWarning,
            stacklevel=3,
        )


def coalignment(reference_map, target_map, method):
    """
    Performs image coalignment using a specified method. It updates the
    metadata of the target map so as to align it with the reference map.

    Parameters
    ----------
    reference_map : sunpy.map.Map
        The reference map to which the target map is to be coaligned.
    target_map : sunpy.map.Map
        The target map to be coaligned to the reference map.
    method : str
        The name of the registered coalignment method to use.

    Returns
    -------
    sunpy.map.Map
        The coaligned target map with the updated metadata.

    Raises
    ------
    ValueError
        If the specified method is not registered.
    """
    if method not in registered_methods:
        msg = f"Method {method} is not a registered method. Please register before using."
        raise ValueError(msg)
    target_array = target_map.data
    reference_array = reference_map.data

    # Warn user if any NANs, Infs, etc are present in the input or the template array
    warn_user_of_nan(target_array, "target")
    warn_user_of_nan(reference_array, "reference")

    # Warn user if the separation between the reference and target maps is large
    warn_user_of_separation(reference_map, target_map)

    affine_params = registered_methods[method](reference_array, target_array)
    return update_fits_wcs_metadata(reference_map, target_map, affine_params)
