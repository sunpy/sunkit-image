from typing import NamedTuple

import astropy.units as u
import numpy as np
from skimage.feature import match_template

from sunkit_image.utils.decorators import register_coalignment_method

__all__ = ["match_template_coalign"]


class affine_params(NamedTuple):
    scale: tuple[tuple[float, float], tuple[float, float]]
    rotation: float
    translation: tuple[float, float]


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
    affine_params
        A named tuple containing the following affine transformation parameters:
        - scale: list
            A list of tuples representing the scale transformation as an identity matrix.
        - rotation: float
            The rotation angle in radians, which is fixed at 0.0 in this function.
        - translation: tuple
            A tuple containing the x and y translation values in pixels.
    """
    corr = match_template(np.float64(input_array), np.float64(template_array))

    # Find the best match location
    y_shift, x_shift = _find_best_match_location(corr)
    # Particularly for this, there is no change in the rotation or scaling, hence the hardcoded values of scale to 1.0 & rotation to identity matrix
    scale = [(1.0, 0), (0, 1.0)]
    rotation = 0.0  # Considering the angle is in radians by default
    return affine_params(scale=scale, rotation=rotation, translation=(x_shift * u.pixel, y_shift * u.pixel))
