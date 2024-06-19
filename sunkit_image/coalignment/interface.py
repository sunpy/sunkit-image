import warnings

import numpy as np
import sunpy.map
from sunpy.util.exceptions import SunpyUserWarning

from sunkit_image.coalignment.util.decorators import registered_methods

__all__ = ["coalignment"]


def convert_array_to_map(array_obj, map_obj):
    """
    Convert a 2D numpy array to a sunpy Map object using the header of a given
    map object.

    Parameters
    ----------
    array_obj : `numpy.ndarray`
        The 2D numpy array to be converted.
    map_obj : `sunpy.map.Map`
        The map object whose header is to be used for the new map.

    Returns
    -------
    `sunpy.map.Map`
        A new sunpy map object with the data from `array_obj` and the header from `map_obj`.
    """
    header = map_obj.meta.copy()
    header["crpix1"] -= array_obj.shape[1] / 2.0 - map_obj.data.shape[1] / 2.0
    header["crpix2"] -= array_obj.shape[0] / 2.0 - map_obj.data.shape[0] / 2.0
    return sunpy.map.Map(array_obj, header)


def warn_user_of_nan(array, name):
    """
    Warn the user if there are NaN values in the input array.

    Parameters
    ----------
    array : `numpy.ndarray`
        The input array to be checked for NaN values.
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


def coalignment(reference_map, target_map, method):
    """
    Interface for performing image coalignment using a specified method.

    Parameters
    ----------
    method : str
        The name of the registered coalignment method to use.
    input_map : `sunpy.map.Map`
        The input map to be coaligned.
    template_map : `sunpy.map.Map`
        The template map to which the input map is to be coaligned.
    handle_nan : callable, optional
        Function to handle NaN values in the input and template arrays.

    Returns
    -------
    `sunpy.map.Map`
        The coaligned input map.

    Raises
    ------
    ValueError
        If the specified method is not registered.
    """
    if method not in registered_methods:
        msg = f"Method {method} is not a registered method. Please register before using."
        raise ValueError(msg)
    target_array = np.float64(target_map.data)
    reference_array = np.float64(reference_map.data)

    # Warn user if any NANs, Infs, etc are present in the input or the template array
    warn_user_of_nan(target_array, "target")
    warn_user_of_nan(reference_array, "reference")

    coalign_output = registered_methods[method](target_array, reference_array)

    return convert_array_to_map(coalign_output["coaligned_array"], target_map)
