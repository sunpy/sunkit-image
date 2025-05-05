import warnings
from typing import NamedTuple

import numpy as np

import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.sun.models import differential_rotation
from sunpy.util.exceptions import SunpyUserWarning

__all__ = ["REGISTERED_METHODS", "AffineParams", "register_coalignment_method"]

# Global Dictionary to store the registered methods and their names
REGISTERED_METHODS = {}


class AffineParams(NamedTuple):
    """
    A 2-element tuple containing scale values defining the image scaling along
    the x and y axes.
    """
    scale: tuple[float, float]
    """
    A 2x2 matrix defining the rotation transformation of the image.
    """
    rotation_matrix: tuple[tuple[float, float], tuple[float, float]]
    """
    A 2-element tuple stores the translation of the image along the x and y
    axes.
    """
    translation: tuple[float, float]


def register_coalignment_method(name):
    """
    Registers a coalignment method to be used by the coalignment interface.

    Parameters
    ----------
    name : str
        The name of the coalignment method.
    """

    def decorator(func):
        REGISTERED_METHODS[name] = func
        return func

    return decorator


def _update_fits_wcs_metadata(reference_map, target_map, affine_params):
    """
    Update the metadata of a sunpy.map.Map` based on affine transformation
    parameters.

    Parameters
    ----------
    reference_map : `sunpy.map.Map`
        The reference map object to which the target map is to be coaligned.
    target_map : `sunpy.map.Map`
        The original map object whose metadata is to be updated.
    affine_params : `NamedTuple`
        A `NamedTuple` containing the affine transformation parameters.
        If you want to use a custom object, it must have attributes for "translation" (dx, dy), "scale", and "rotation_matrix".

    Returns
    -------
    `sunpy.map.Map`
        A new sunpy map object with updated metadata reflecting the affine transformation.
    """
    # Updating the PC_ij matrix
    new_pc_matrix = target_map.rotation_matrix @ affine_params.rotation_matrix
    # Calculate the new reference pixel.
    old_reference_pixel = np.asarray([target_map.reference_pixel.x.value, target_map.reference_pixel.y.value])
    new_reference_pixel = affine_params.scale * affine_params.rotation_matrix @ old_reference_pixel + affine_params.translation
    reference_coord = reference_map.wcs.pixel_to_world(new_reference_pixel[0],new_reference_pixel[1])
    Txshift = reference_coord.Tx - target_map.reference_coordinate.Tx
    Tyshift = reference_coord.Ty - target_map.reference_coordinate.Ty

    # Create a new map with the updated metadata
    fixed_map = target_map.shift_reference_coord(Txshift, Tyshift)
    fixed_map.meta["PC1_1"] = new_pc_matrix[0, 0]
    fixed_map.meta["PC1_2"] = new_pc_matrix[0, 1]
    fixed_map.meta["PC2_1"] = new_pc_matrix[1, 0]
    fixed_map.meta["PC2_2"] = new_pc_matrix[1, 1]
    fixed_map.meta['cdelt1'] = (target_map.scale[0] / affine_params.scale[0]).value
    fixed_map.meta['cdelt2'] = (target_map.scale[1] / affine_params.scale[1]).value
    return fixed_map


def _warn_user_of_separation(reference_map, target_map):
    """
    Issues a warning if the separation between the ``reference_map`` and
    ``target_map`` is "large".

    Parameters
    ----------
    reference_map : `sunpy.map.Map`
        The reference map to which the target map is to be coaligned.
    target_map : `sunpy.map.Map`
        The target map to be coaligned to the reference map.
    """
    # Maximum angular separation allowed between the reference and target maps
    tolerable_angular_separation = 1*u.deg
    ref_coord = SkyCoord(reference_map.observer_coordinate)
    target_coord = SkyCoord(target_map.observer_coordinate)
    if astropy.__version__ >= "6.1.0":
        angular_separation = ref_coord.separation(target_coord, origin_mismatch="ignore")
    else:
        angular_separation = ref_coord.separation(target_coord)
    if angular_separation > tolerable_angular_separation:
        warnings.warn(
            "The angular separation between the reference and target maps is large. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure the maps are close in space.",
            SunpyUserWarning,
            stacklevel=3,
        )
    # Calculate time difference and convert to separation angle
    ref_time = reference_map.date
    target_time = target_map.date
    time_diff = np.abs(ref_time - target_time)
    time_separation_angle = differential_rotation(time_diff.to(u.day), reference_map.center.Tx, model='howard')
    if time_separation_angle > tolerable_angular_separation:
        warnings.warn(
            "The time difference between the reference and target maps in time is large. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure the maps are close in time.",
            SunpyUserWarning,
            stacklevel=3,
        )


def coalign(reference_map, target_map, method='match_template', **kwargs):
    """
    Performs image coalignment using the specified method.

    This function updates the metadata of the target map to align it with the reference map.

    .. note::

        * This function is intended to correct maps with known incorrect metadata.
          It is not designed to address issues like differential rotation or changes in observer location, which are encoded in the coordinate metadata.
        * The function modifies the metadata of the map, not the underlying array data.
          For adjustments that involve coordinate transformations, consider using `~sunpy.map.GenericMap.reproject_to` instead.

    Parameters
    ----------
    reference_map : `sunpy.map.Map`
        The reference map to which the target map is to be coaligned.
    target_map : `sunpy.map.Map`
        The target map to be coaligned to the reference map.
    method : {{{coalignment_function_names}}}, optional
        The name of the registered coalignment method to use.
        Defaults to 'match_template'.
    kwargs : `dict`
        Additional keyword arguments to pass to the registered method.

    Returns
    -------
    `sunpy.map.Map`
        The coaligned target map with the updated metadata.

    Raises
    ------
    ValueError
        If the specified method is not registered.
    """
    if method not in REGISTERED_METHODS:
        msg = (f"Method {method} is not a registered method: {','.join(REGISTERED_METHODS.keys())}. "
        "The method needs to be registered, with the correct import.")
        raise ValueError(msg)
    _warn_user_of_separation(reference_map, target_map)
    affine_params = REGISTERED_METHODS[method](reference_map.data, target_map.data, **kwargs)
    return _update_fits_wcs_metadata(reference_map, target_map, affine_params)


# Generate the string with allowable coalignment-function names for use in docstrings
_coalignment_function_names = ", ".join([f"``'{name}'``" for name in REGISTERED_METHODS])
# Insert into the docstring for coalign. We cannot use the add_common_docstring decorator
# due to what would be a circular loop in definitions.
coalign.__doc__ = coalign.__doc__.format(coalignment_function_names=_coalignment_function_names)  # type: ignore
