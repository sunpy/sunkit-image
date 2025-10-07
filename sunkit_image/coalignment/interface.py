import warnings
from typing import NamedTuple

import numpy as np

import astropy
import astropy.units as u

from sunpy import log
from sunpy.sun.models import differential_rotation
from sunpy.util.exceptions import SunpyUserWarning

__all__ = [
    "REGISTERED_METHODS",
    "AffineParams",
    "coalign",
    "register_coalignment_method",
]

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


def _update_fits_wcs_metadata(target_map, reference_map, affine_params):
    """
    Update the metadata of a sunpy.map.Map` based on affine transformation
    parameters.

    Parameters
    ----------
    target_map : `sunpy.map.Map`
        The original map object whose metadata is to be updated.
    reference_map : `sunpy.map.Map`
        The reference map object to which the target map is to be coaligned.
    affine_params : `NamedTuple`
        A `NamedTuple` containing the affine transformation parameters.
        If you want to use a custom object, it must have attributes for "translation", "scale", and "rotation_matrix".

    Returns
    -------
    `sunpy.map.Map`
        A new sunpy map object with updated metadata reflecting the affine transformation.
    """
    # NOTE: Currently, the only metadata updates that are currently supported are shifts in
    # the reference coordinate. Once other updates are supported, this check can be removed.
    if not (affine_params.rotation_matrix == np.eye(2)).all():
        raise NotImplementedError('Changes to the rotation metadata are currently not supported.')
    if not (affine_params.scale == np.array([1,1])).all():
        raise NotImplementedError('Changes to the pixel scale metadata are currently not supported.')
    # Calculate the new reference pixel.
    old_reference_pixel = u.Quantity(target_map.reference_pixel).to_value('pixel')
    new_reference_pixel = affine_params.scale * affine_params.rotation_matrix @ old_reference_pixel + affine_params.translation
    new_reference_coordinate = reference_map.wcs.pixel_to_world(*new_reference_pixel)
    # Create a new map with the updated metadata
    log.debug(f"Shifting reference coordinate from {target_map.reference_coordinate} to {new_reference_coordinate} by {new_reference_coordinate.Tx - target_map.reference_coordinate.Tx}, {new_reference_coordinate.Ty - target_map.reference_coordinate.Ty}")
    return target_map.shift_reference_coord(
        new_reference_coordinate.Tx - target_map.reference_coordinate.Tx,
        new_reference_coordinate.Ty - target_map.reference_coordinate.Ty,
    )


def _warn_user_of_separation(target_map, reference_map):
    """
    Issues a warning if the separation between the ``reference_map`` and
    ``target_map`` is "large".

    Parameters
    ----------
    target_map : `sunpy.map.Map`
        The target map to be coaligned to the reference map.
    reference_map : `sunpy.map.Map`
        The reference map to which the target map is to be coaligned.
    """
    # Maximum angular separation allowed between the reference and target maps
    tolerable_angular_separation = 1*u.deg
    if astropy.__version__ >= "6.1.0":
        angular_separation = reference_map.observer_coordinate.separation(
            target_map.observer_coordinate, origin_mismatch="ignore"
        )
    else:
        angular_separation = reference_map.observer_coordinate.separation(
            target_map.observer_coordinate
        )
    if angular_separation > tolerable_angular_separation:
        warnings.warn(
            "The angular separation between the reference and target maps is large. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure the maps are close in space.",
            SunpyUserWarning,
            stacklevel=3,
        )
    # Calculate time difference and convert to separation angle
    time_diff = np.abs((reference_map.date - target_map.date).to('s'))
    time_separation_angle = differential_rotation(
        time_diff,
        reference_map.center.transform_to('heliographic_stonyhurst').lat,
        model='howard'
    )
    if time_separation_angle > tolerable_angular_separation:
        warnings.warn(
            "The time difference between the reference and target maps in time is large. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure the maps are close in time.",
            SunpyUserWarning,
            stacklevel=3,
        )


def coalign(target_map, reference_map, method='match_template', **kwargs):
    """
    Performs image coalignment using the specified method by updating the metadata of the
    target map to align it with the reference map.

    .. note::

        This function is intended to correct maps with known incorrect metadata.
        It is not designed to address issues like differential rotation or changes in observer
        location, which are encoded in the coordinate metadata.

    .. note::

        This function modifies the metadata of the map, not the underlying array data.
        For adjustments that involve coordinate transformations, consider using
        `~sunpy.map.GenericMap.reproject_to` instead.

    Parameters
    ----------
    target_map : `sunpy.map.Map`
        The map to be coaligned. The target map should be fully contained within the reference map and,
        for best results, have approximately the same observer location. For coalignment methods which do
        not account for different pixel scales or rotations, it is recommended that the reference
        map and target map are resampled and/or rotated such that they have the same orientation and
        resolution.
    reference_map : `sunpy.map.Map`
        The map to which the target map is to be coaligned. For best results, the pointing data of this
        map should be well-known.
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
    _warn_user_of_separation(target_map, reference_map)
    affine_params = REGISTERED_METHODS[method](target_map.data, reference_map.data, **kwargs)
    return _update_fits_wcs_metadata(target_map, reference_map, affine_params)


# Generate the string with allowable coalignment-function names for use in docstrings
_coalignment_function_names = ", ".join([f"``'{name}'``" for name in REGISTERED_METHODS])
# Insert into the docstring for coalign. We cannot use the add_common_docstring decorator
# due to what would be a circular loop in definitions.
coalign.__doc__ = coalign.__doc__.format(coalignment_function_names=_coalignment_function_names)  # TYPE: IGNORE
