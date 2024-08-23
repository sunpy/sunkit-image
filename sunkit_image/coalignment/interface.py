import warnings
from typing import NamedTuple
import numpy as np

import astropy.units as u
from astropy.coordinates import SkyCoord

from sunpy.util.exceptions import SunpyUserWarning
from sunpy.sun.models import differential_rotation

from sunkit_image.coalignment.decorators import registered_methods

__all__ = ["AffineParams"]


class AffineParams(NamedTuple):
    """
    A 2-element tuple containing scale values defining the image scaling along the x and y axes.
    """
    scale: tuple[float, float]
    """
    A 2x2 matrix defining the rotation transformation of the image.
    """
    rotation_matrix: tuple[tuple[float, float], tuple[float, float]]
    """
    A 2-element tuple stores the translation of the image along the x and y axes.
    """
    translation: tuple[float, float]


def update_fits_wcs_metadata(reference_map, target_map, affine_params):
    """
    Update the metadata of a sunpy Map object based on affine transformation
    parameters.

    Parameters
    ----------
    reference_map : `sunpy.map.Map`
        The reference map object to which the target map is to be coaligned.
    target_map : `sunpy.map.Map`
        The original map object whose metadata is to be updated.
    AffineParams : object
        An object containing the affine transformation parameters. This object must
        have attributes for translation (dx, dy), scale, and rotation.

    Returns
    -------
    `sunpy.map.Map`
        A new sunpy map object with updated metadata reflecting the affine transformation.
    """
    # Extacting the affine parameters
    pc_matrix = target_map.rotation_matrix
    translation = AffineParams.translation
    scale = AffineParams.scale
    rotation_matrix = AffineParams.rotation_matrix
    # Updating the PC matrix
    new_pc_matrix = pc_matrix @ rotation_matrix
    # Calculate the new reference pixel. 
    old_reference_pixel = np.array([target_map.reference_pixel.x.value, target_map.reference_pixel.y.value])
    new_reference_pixel = scale*rotation_matrix @ old_reference_pixel + translation
    reference_coord = reference_map.wcs.pixel_to_world(new_reference_pixel[0],new_reference_pixel[1])
    Txshift = reference_coord.Tx - target_map.reference_coordinate.Tx
    Tyshift = reference_coord.Ty - target_map.reference_coordinate.Ty
    # Create a new map with the updated metadata
    fixed_map = target_map.shift_reference_coord(Txshift, Tyshift)

    fixed_map.meta["PC1_1"] = new_pc_matrix[0, 0]
    fixed_map.meta["PC1_2"] = new_pc_matrix[0, 1]
    fixed_map.meta["PC2_1"] = new_pc_matrix[1, 0]
    fixed_map.meta["PC2_2"] = new_pc_matrix[1, 1]

    fixed_map.meta['cdelt1'] = (target_map.scale[0] / scale[0]).value
    fixed_map.meta['cdelt2'] = (target_map.scale[1] / scale[1]).value

    return fixed_map


def warn_user_of_separation(reference_map,target_map):
    """
    Issues a warning if the separation between the reference and target maps is
    large.

    Parameters
    ----------
    reference_map : `sunpy.map.Map`
        The reference map to which the target map is to be coaligned.
    target_map : `sunpy.map.Map`
        The target map to be coaligned to the reference map.
    """
    ref_coord = SkyCoord(reference_map.observer_coordinate)
    target_coord = SkyCoord(target_map.observer_coordinate)
    angular_separation = ref_coord.separation(target_coord, origin_mismatch="ignore")
    if angular_separation > (1*u.deg):
        warnings.warn(## warn due to large angular separation
            "The angular separation between the reference and target maps is large. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure the maps are close in space.",
            SunpyUserWarning,
            stacklevel=3,
        )
    # Calculate time difference and convert to separation angle
    ref_time = reference_map.date
    target_time = target_map.date
    time_diff = abs(ref_time - target_time)
    time_separation_angle = differential_rotation(time_diff.to(u.day), reference_map.center.Tx, model='howard')
    if time_separation_angle > (1*u.deg):
        warnings.warn(## warn due to large time separation
            "The time difference between the reference and target maps in time is large. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure the maps are close in time.",
            SunpyUserWarning,
            stacklevel=3,
        )


def coalign(reference_map, target_map, method='match_template'):
    """
    Performs image coalignment using a specified method (defaults to `~sunkit_image.coalignment.match_template.match_template_coalign`).
    This function updates the metadata of the target map to align it with the reference map.

    .. note::

        * This function is intended to correct maps with known incorrect metadata. It is not designed to address issues like differential rotation or changes in observer location, which are encoded in the coordinate metadata.
        * The function modifies the metadata of the map, not the underlying array data. For adjustments that involve coordinate transformations, consider using `~sunpy.map.GenericMap.reproject_to` instead.
    
    Parameters
    ----------
    reference_map : `sunpy.map.Map`
        The reference map to which the target map is to be coaligned.
    target_map : `sunpy.map.Map`
        The target map to be coaligned to the reference map.
    method : `str`
        The name of the registered coalignment method to use.

    Returns
    -------
    `sunpy.map.Map`
        The coaligned target map with the updated metadata.

    Raises
    ------
    ValueError
        If the specified method is not registered.
    """
    if method not in registered_methods:
        msg = (f"Method {method} is not a registered method: {list(registered_methods.keys())}."
        "The method needs to be registered, with the correct import.")
        raise ValueError(msg)
    target_array = target_map.data
    reference_array = reference_map.data

    warn_user_of_separation(reference_map, target_map)

    AffineParams = registered_methods[method](reference_array, target_array)
    return update_fits_wcs_metadata(reference_map, target_map, AffineParams)
