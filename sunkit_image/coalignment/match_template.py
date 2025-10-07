import numpy as np
from skimage.feature import match_template

from sunpy import log

from sunkit_image.coalignment.register import register_coalignment_method

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
def match_template_coalign(target_array, reference_array, **kwargs):
    """
    Perform coalignment by matching the input array to the target array.

    Parameters
    ----------
    target_array : `numpy.ndarray`
        The input 2D array to be coaligned.
    reference_array : `numpy.ndarray`
        The template 2D array to align to.
    kwargs: dict
        Passed to `skimage.feature.match_template`.

    Returns
    -------
    `sunkit_image.coalignment.interface.AffineParams`
        A `NamedTuple` containing the following affine transformation parameters:

        - scale : `list`
            A list of tuples representing the scale transformation as an identity matrix.
        - rotation : `float`
            The rotation angle in radians, which is fixed at 0.0 in this function.
        - translation : `tuple`
            A tuple containing the x and y translation values.
    """
    from sunkit_image.coalignment.interface import AffineParams  # NOQA: PLC0415

    corr = match_template(np.float64(reference_array), np.float64(target_array), **kwargs)
    # TODO: Work out what is going on
    if corr.ndim != target_array.ndim:
        raise ValueError("The correlation output failed to work out a match.")
    # Find the best match location
    y_shift, x_shift = _find_best_match_location(corr)
    log.debug(f"Match template shift: x: {x_shift}, y: {y_shift}")
    # Particularly for this method, there is no change in the rotation or scaling,
    # hence the hardcoded values of scale to 1.0 & rotation to identity matrix
    scale = np.array([1.0, 1.0])
    rotation_matrix = np.eye(2)
    return AffineParams(scale=scale, rotation_matrix=rotation_matrix, translation=(x_shift, y_shift))
