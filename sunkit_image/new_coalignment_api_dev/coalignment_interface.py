import warnings

import astropy.units as u
import numpy as np
import sunpy.map
from skimage.feature import match_template
from sunpy.util.exceptions import SunpyUserWarning


############################ Coalignment Interface begins #################################
@u.quantity_input
def _clip_edges(data, yclips: u.pix, xclips: u.pix):
    """
    Clips off the "y" and "x" edges of a 2D array according to a list of pixel
    values. This function is useful for removing data at the edge of 2d images
    that may be affected by shifts from solar de- rotation and layer co-
    registration, leaving an image unaffected by edge effects.

    Parameters
    ----------
    data : `numpy.ndarray`
        A numpy array of shape ``(ny, nx)``.
    yclips : `astropy.units.Quantity`
        The amount to clip in the y-direction of the data. Has units of
        pixels, and values should be whole non-negative numbers.
    xclips : `astropy.units.Quantity`
        The amount to clip in the x-direction of the data. Has units of
        pixels, and values should be whole non-negative numbers.

    Returns
    -------
    `numpy.ndarray`
        A 2D image with edges clipped off according to ``yclips`` and ``xclips``
        arrays.
    """
    ny = data.shape[0]
    nx = data.shape[1]
    # The purpose of the int below is to ensure integer type since by default
    # astropy quantities are converted to floats.
    return data[int(yclips[0].value) : ny - int(yclips[1].value), int(xclips[0].value) : nx - int(xclips[1].value)]


@u.quantity_input
def _calculate_clipping(y: u.pix, x: u.pix):
    """
    Return the upper and lower clipping values for the "y" and "x" directions.

    Parameters
    ----------
    y : `astropy.units.Quantity`
        An array of pixel shifts in the y-direction for an image.
    x : `astropy.units.Quantity`
        An array of pixel shifts in the x-direction for an image.

    Returns
    -------
    `tuple`
        The tuple is of the form ``([y0, y1], [x0, x1])``.
        The number of (integer) pixels that need to be clipped off at each
        edge in an image. The first element in the tuple is a list that gives
        the number of pixels to clip in the y-direction. The first element in
        that list is the number of rows to clip at the lower edge of the image
        in y. The clipped image has "clipping[0][0]" rows removed from its
        lower edge when compared to the original image. The second element in
        that list is the number of rows to clip at the upper edge of the image
        in y. The clipped image has "clipping[0][1]" rows removed from its
        upper edge when compared to the original image. The second element in
        the "clipping" tuple applies similarly to the x-direction (image
        columns). The parameters ``y0, y1, x0, x1`` have the type
        `~astropy.units.Quantity`.
    """
    return (
        [_lower_clip(y.value), _upper_clip(y.value)] * u.pix,
        [_lower_clip(x.value), _upper_clip(x.value)] * u.pix,
    )


def _upper_clip(z):
    """
    Find smallest integer bigger than all the positive entries in the input
    array.
    """
    zupper = 0
    zcond = z >= 0
    if np.any(zcond):
        zupper = int(np.max(np.ceil(z[zcond])))
    return zupper


def _lower_clip(z):
    """
    Find smallest positive integer bigger than the absolute values of the
    negative entries in the input array.
    """
    zlower = 0
    zcond = z <= 0
    if np.any(zcond):
        zlower = int(np.max(np.ceil(-z[zcond])))
    return zlower


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


def coalignment_interface(method, input_map, template_map, handle_nan=None):
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
    input_array = np.float64(input_map.data)
    template_array = np.float64(template_map.data)

    # Warn user if any NANs, Infs, etc are present in the input or the template array
    if not np.all(np.isfinite(input_array)):
        if not handle_nan:
            warnings.warn(
                "The layer image has nonfinite entries. "
                "This could cause errors when calculating shift between two "
                "images. Please make sure there are no infinity or "
                "Not a Number values. For instance, replacing them with a "
                "local mean.",
                SunpyUserWarning,
                stacklevel=3,
            )
        else:
            input_array = handle_nan(input_array)

    if not np.all(np.isfinite(template_array)):
        if not handle_nan:
            warnings.warn(
                "The template image has nonfinite entries. "
                "This could cause errors when calculating shift between two "
                "images. Please make sure there are no infinity or "
                "Not a Number values. For instance, replacing them with a "
                "local mean.",
                SunpyUserWarning,
                stacklevel=3,
            )
        else:
            template_array = handle_nan(template_array)

    shifts = registered_methods[method](input_array, template_array)
    # Calculate the clipping required
    yclips, xclips = _calculate_clipping(shifts["x"] * u.pix, shifts["y"] * u.pix)
    # Clip 'em
    coaligned_input_array = _clip_edges(input_array, yclips, xclips)
    return convert_array_to_map(coaligned_input_array, input_map)

######################################## Coalignment interface ends ##################
