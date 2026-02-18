import warnings
from dataclasses import dataclass

import numpy as np

import astropy
import astropy.units as u

from sunpy import log
from sunpy.sun.models import differential_rotation
from sunpy.util.exceptions import SunpyUserWarning

__all__ = [
    "AffineParams",
    "coalign_map",
    "update_map_metadata",
]


@dataclass
class AffineParams:
    """
    A 2-element tuple containing scale values defining the image scaling along
    the x and y axes.
    """
    scale: tuple[float, float] | list[float] | np.ndarray
    """
    A 2x2 matrix defining the rotation transformation of the image.
    """
    rotation_matrix: tuple[tuple[float, float], tuple[float, float]] | np.ndarray
    """
    A 2-element tuple stores the translation of the image along the x and y
    axes.
    """
    translation: tuple[float, float] | list[float] | np.ndarray


def update_map_metadata(target_map, reference_map, affine_params):
    """
    Update the metadata of a sunpy.map.Map` based on affine transformation
    parameters.

    .. warning::

        This function is currently only designed to update the reference coordinate
        metadata based on the translation component of the affine transformation.
        Changes to the rotation and scale metadata are not currently supported.
        If you have a use case that requires changes to the rotation or scale metadata
        `please open an issue on the issue tracker <https://github.com/sunpy/sunkit-image/issues>`__.

    Parameters
    ----------
    target_map : `sunpy.map.Map`
        The original map object whose metadata is to be updated.
    reference_map : `sunpy.map.Map`
        The reference map object to which the target map is to be coaligned.
    affine_params : `NamedTuple`
        A `NamedTuple` containing the affine transformation parameters.

    Returns
    -------
    `sunpy.map.Map`
        A new sunpy map object with updated metadata reflecting the affine transformation.
    """
    # NOTE: Currently, the only metadata updates that are supported are shifts in
    # the reference coordinate. Once other updates are supported, this check can be removed.
    full_msg = (
        "\nIf you have a use case that requires changes to the rotation or scale metadata, "
        "please open an issue at https://github.com/sunpy/sunkit-image/issues with details "
        "about your use case and the type of metadata changes you require."
    )
    if not (affine_params.rotation_matrix == np.eye(2)).all():
        raise NotImplementedError('Changes to the rotation metadata are currently not supported.'+ full_msg)
    if not (affine_params.scale == np.array([1,1])).all():
        raise NotImplementedError('Changes to the pixel scale metadata are currently not supported.'+ full_msg)
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
            "The angular separation between the observer coordinates of "
            "the reference and target maps is large. This can lead to spurious "
            "results when calculating shift between two images. Consider choosing "
            "a reference map with a similar observer position to your target map.",
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
            "The difference in observation times of the reference and target maps is large. "
            "This can lead to spurious results when calculating shifts between two images."
            "Consider choosing a reference map with a similar observation time to your target map.",
            SunpyUserWarning,
            stacklevel=3,
        )


def _warn_user_of_plate_scale_difference(target_map, reference_map):
    """
    Issues a warning if there is a plate scale difference between the
    ``reference_map`` and ``target_map``.

    Parameters
    ----------
    target_map : `sunpy.map.Map`
        The target map to be coaligned to the reference map.
    reference_map : `sunpy.map.Map`
        The reference map to which the target map is to be coaligned.
    """
    if not u.allclose(target_map.scale, reference_map.scale):
        warnings.warn(
            "The reference and target maps have different plate scales. "
            "This can lead to spurious results when calculating the shift between two arrays. "
            "Consider resampling the reference map to have the same plate scale as the target map.",
            SunpyUserWarning,
            stacklevel=3,
        )


def coalign_map(target_map, reference_map, method='match_template', **kwargs):
    """
    Performs image coalignment using the specified method by updating the metadata of the
    target map to align it with the reference map.

    .. note::

        This function is intended to correct maps with inaccurate metadata.
        It is not designed to correct for differential rotation or changes in observer
        location.

    .. warning::

        This function is currently only designed to update the reference coordinate
        metadata based on the translation component of the affine transformation.
        Changes to the rotation and scale metadata are not currently supported.
        If you have a use case that requires changes to the rotation or scale metadata
        `please open an issue on the issue tracker <https://github.com/sunpy/sunkit-image/issues>`__.

    .. note::

        This function modifies the metadata of the map, not the underlying array data.
        For adjustments that involve coordinate transformations, consider using
        `~sunpy.map.GenericMap.reproject_to` instead.

    Parameters
    ----------
    target_map : `sunpy.map.GenericMap`
        The map to be coaligned.
    reference_map : `sunpy.map.GenericMap`
        The map to which the target map is to be coaligned. It is expected that the pointing data of this map
        is accurate. For best results, ``reference_map`` and ``target_map`` should have approximately the
        same observer location and observation time. For coalignment methods which do
        not account for different pixel scales or rotations, it is recommended that ``reference_map``
        and ``target_map`` are resampled and/or rotated such that they have the same orientation and
        plate scale.
    method : {{{coalignment_function_names}}}, optional
        The name of the registered coalignment method to use.
        Defaults to `~sunkit_image.coalignment.match_template.match_template_coalign`.
    kwargs : `dict`
        Additional keyword arguments to pass to the registered method.

    Returns
    -------
    `sunpy.map.GenericMap`
        The coaligned target map with the updated metadata.

    Raises
    ------
    ValueError
        If the specified method is not registered.
    """
    if method not in REGISTERED_METHODS:
        msg = (f"Method {method} is not a registered method: {','.join(REGISTERED_METHODS.keys())}. "
        "If you expect this method to be present, ensure the method has been registered with the correct import.")
        raise ValueError(msg)
    _warn_user_of_separation(target_map, reference_map)
    _warn_user_of_plate_scale_difference(target_map, reference_map)

    affine_params = REGISTERED_METHODS[method](target_map.data, reference_map.data, **kwargs)
    return update_map_metadata(target_map, reference_map, affine_params)


from sunkit_image.coalignment.register import REGISTERED_METHODS  # isort:skip # NOQA: E402
# Generate the string with allowable coalignment-function names for use in docstrings
_coalignment_function_names = ", ".join([f"``'{name}'``" for name in REGISTERED_METHODS])
# Insert into the docstring for coalign_map. We cannot use the add_common_docstring decorator
# due to what would be a circular loop in definitions.
coalign_map.__doc__ = coalign_map.__doc__.format(coalignment_function_names=_coalignment_function_names)  # type: ignore
