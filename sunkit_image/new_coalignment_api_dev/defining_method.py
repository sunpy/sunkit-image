import warnings

import astropy.units as u
import numpy as np
import sunpy.map
from skimage.feature import match_template
from sunpy.util.exceptions import SunpyUserWarning

######################################## Defining a method ###########################
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

    # Apply the shift to get the coaligned input array
    return {"x": x_shift, "y": y_shift}


################################ Registering the defined method ########################
register_coalignment_method("match_template", match_template_coalign)