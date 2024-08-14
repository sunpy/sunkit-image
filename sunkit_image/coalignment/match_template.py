import numpy as np
from skimage.feature import match_template

import astropy.units as u

from sunkit_image.coalignment.decorators import register_coalignment_method
from sunkit_image.coalignment.interface import affine_params

__all__ = ["match_template_coalign"]


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
    return (-0.5 * y.dot([-1, 0, 1]))/ y.dot([1, -2, 1])


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
def match_template_coalign(reference_array, target_array):
    """
    Perform coalignment by matching the template array to the input array.

    Parameters
    ----------
    input_array : numpy.ndarray
        The input 2D array to be coaligned.
    template_array : numpy.ndarray
        The template 2D array to align to.

    Returns
    -------
    affine_params
        A named tuple containing the following affine transformation parameters:
        - scale: list
            A list of tuples representing the scale transformation as an identity matrix.
        - rotation: float
            The rotation angle in radians, which is fixed at 0.0 in this function.
        - translation: tuple
            A tuple containing the x and y translation values in pixels.
    """
    corr = match_template(np.float64(reference_array), np.float64(target_array))
    # Find the best match location
    y_shift, x_shift = _find_best_match_location(corr)
    # Particularly for this method, there is no change in the rotation or scaling, hence the hardcoded values of scale to 1.0 & rotation to identity matrix
    scale = np.array([1.0, 1.0])
    rotation_matrix = np.eye(2)
    return affine_params(scale=scale, rotation_matrix=rotation_matrix, translation=(x_shift * u.pixel, y_shift * u.pixel))
