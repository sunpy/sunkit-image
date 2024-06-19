import astropy.units as u
import numpy as np
from skimage.feature import match_template

from sunkit_image.coalignment.util.decorators import register_coalignment_method

__all__ = ["match_template_coalign"]


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


def _parabolic_turning_point(y):
    """
    Calculate the turning point of a parabola given three points.

    Parameters
    ----------
    y : `numpy.ndarray`
        An array of three points defining the parabola.

    Returns
    -------
    float
        The x-coordinate of the turning point.
    """
    numerator = -0.5 * y.dot([-1, 0, 1])
    denominator = y.dot([1, -2, 1])
    return numerator / denominator


def _get_correlation_shifts(array):
    """
    Calculate the shifts in x and y directions based on the correlation array.

    Parameters
    ----------
    array : `numpy.ndarray`
        A 2D array representing the correlation values.

    Returns
    -------
    tuple
        The shifts in y and x directions.

    Raises
    ------
    ValueError
        If the input array dimensions are greater than 3 in any direction.
    """
    ny, nx = array.shape
    if nx > 3 or ny > 3:
        msg = "Input array dimension should not be greater than 3 in any dimension."
        raise ValueError(msg)

    ij = np.unravel_index(np.argmax(array), array.shape)
    x_max_location, y_max_location = ij[::-1]

    y_location = _parabolic_turning_point(array[:, x_max_location]) if ny == 3 else 1.0 * y_max_location
    x_location = _parabolic_turning_point(array[y_max_location, :]) if nx == 3 else 1.0 * x_max_location

    return y_location, x_location


def _find_best_match_location(corr):
    """
    Find the best match location in the correlation array.

    Parameters
    ----------
    corr : `numpy.ndarray`
        The correlation array.

    Returns
    -------
    tuple
        The best match location in the y and x directions.
    """
    ij = np.unravel_index(np.argmax(corr), corr.shape)
    cor_max_x, cor_max_y = ij[::-1]

    array_maximum = corr[
        max(0, cor_max_y - 1) : min(cor_max_y + 2, corr.shape[0] - 1),
        max(0, cor_max_x - 1) : min(cor_max_x + 2, corr.shape[1] - 1),
    ]

    y_shift_maximum, x_shift_maximum = _get_correlation_shifts(array_maximum)

    y_shift_correlation_array = y_shift_maximum + cor_max_y
    x_shift_correlation_array = x_shift_maximum + cor_max_x

    return y_shift_correlation_array, x_shift_correlation_array


@register_coalignment_method("match_template")
def match_template_coalign(input_array, template_array):
    """
    Perform coalignment by matching the template array to the input array.

    Parameters
    ----------
    input_array : `numpy.ndarray`
        The input 2D array to be coaligned.
    template_array : `numpy.ndarray`
        The template 2D array to align to.

    Returns
    -------
    dict
        A dictionary containing the shifts in x and y directions.
    """
    corr = match_template(input_array, template_array)

    # Find the best match location
    y_shift, x_shift = _find_best_match_location(corr)
    # Calculate the clipping required
    yclips, xclips = _calculate_clipping(x_shift * u.pix, y_shift * u.pix)
    # Clip 'em
    coaligned_target_array = _clip_edges(input_array, yclips, xclips)
    # Apply the shift to get the coaligned input array
    return {"shifts": {"x": x_shift, "y": y_shift}, "coaligned_array": coaligned_target_array}
