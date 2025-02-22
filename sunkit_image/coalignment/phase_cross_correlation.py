import numpy as np
from skimage.registration import phase_cross_correlation

from sunkit_image.coalignment.interface import AffineParams, register_coalignment_method

__all__ = ["phase_cross_correlation_coalign"]


@register_coalignment_method("phase_cross_correlation")
def phase_cross_correlation_coalign(input_array, target_array, **kwargs):
    """
    Perform coalignment by phase cross correlation input array to the target array.

    This requires both the input and target arrays to be the same size.

    Parameters
    ----------
    input_array : `numpy.ndarray`
        The input 2D array to be coaligned.
    target_array : `numpy.ndarray`
        The template 2D array to align to.
    kwargs : dict
        Passed to `skimage.registration.phase_cross_correlation`.

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
    if input_array.shape != target_array.shape:
        raise ValueError("Input and target arrays must be the same shape.")
    shift, _, _ = phase_cross_correlation(target_array, input_array, **kwargs)
    # TODO: Check if the shift is correct
    x_shift = shift[1]
    y_shift = shift[0]
    # Particularly for this method, there is no change in the rotation or scaling,
    # hence the hardcoded values of scale to 1.0 & rotation to identity matrix
    scale = np.array([1.0, 1.0])
    rotation_matrix = np.eye(2)
    return AffineParams(scale=scale, rotation_matrix=rotation_matrix, translation=(x_shift, y_shift))
