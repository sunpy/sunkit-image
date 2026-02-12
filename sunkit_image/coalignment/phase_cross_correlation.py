import numpy as np
from skimage.registration import phase_cross_correlation

from sunpy import log

from sunkit_image.coalignment.register import register_coalignment_method

__all__ = ["phase_cross_correlation_coalign"]


@register_coalignment_method("phase_cross_correlation")
def phase_cross_correlation_coalign(target_array, reference_array, **kwargs):
    """
    Perform coalignment by phase cross correlation input array to the target array.

    .. note:: This requires both the input and target arrays to be the same size.

    Coalign ``target_array`` to ``reference_array`` using phase cross-correlation
    via `skimage.registration.phase_cross_correlation`. For more details on this approach,
    please check the documentation of that function including the available keyword
    arguments and the details of the algorithm.

    Parameters
    ----------
    target_array : `numpy.ndarray`
        The input 2D array to be coaligned.
    reference_array : `numpy.ndarray`
        The template 2D array to align to.
    kwargs : dict
        Passed to `skimage.registration.phase_cross_correlation`.

    Returns
    -------
    `sunkit_image.coalignment.interface.AffineParams`
        This method only returns a translation. The scale and rotation
        parameters are unity.
    """
    from sunkit_image.coalignment.interface import AffineParams  # NOQA: PLC0415

    if target_array.shape != reference_array.shape:
        raise ValueError("Input and target arrays must be the same shape.")
    shift, _, _ = phase_cross_correlation(reference_array, target_array, **kwargs)
    # Shift has axis ordering which is consistent with the axis order of the input arrays, so y, x
    # See the example here:
    # https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_translation.html
    x_shift, y_shift = shift[1], shift[0]
    log.debug(f"Phase cross correlation shift: x: {x_shift}, y: {y_shift}")
    # Particularly for this method, there is no change in the rotation or scaling,
    # hence the hardcoded values of scale to 1.0 & rotation to identity matrix
    scale = np.array([1.0, 1.0])
    rotation_matrix = np.eye(2)
    return AffineParams(scale=scale, rotation_matrix=rotation_matrix, translation=(x_shift, y_shift))
