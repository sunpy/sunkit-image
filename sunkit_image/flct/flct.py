import numpy as np

from sunkit_image.flct.utils import column_row_of_two

try:
    from sunkit_image.flct import _pyflct
except ImportError:
    _pyflct = None

__all__ = ["flct"]


def flct(
    image1,
    image2,
    deltat,
    deltas,
    sigma,
    order="row",
    quiet=False,
    biascor=False,
    thresh=0.0,
    absflag=False,
    skip=None,
    xoff=0,
    yoff=0,
    interp=False,
    kr=None,
    pc=False,
    latmin=0,
    latmax=0.2,
):
    """
    Performs Fourier Local Correlation Tracking by calling the FLCT C library.

    .. note::

        * In the references there are some dat files which can be used to test the FLCT code. The
          best method to read those dat files is the `sunkit_image.flct.read_2_images` and
          `sunkit_image.flct.read_3_images` as the arrays would automatically be read in row major
          format.
        * If you use the IDL IO routines to get the input arrays from ``dat`` files,
          the IDL routines always read the binary files in the column major, but both Python and C,
          on which these functions are written are row major so the order of the arrays have to be
          changed which can be done with the ``order`` keyword. This may lead to different values
          in both the cases.
        * If your input arrays are column major then pass the `order` parameter as `column` and it
          will automatically take care of the order change. But this can produce some changes in
          the values of the arrays.

    .. warning::

        All the below limitations have been directly taken from the C source user manual without any modifications.
        The original user manual can be found `here <http://solarmuri.ssl.berkeley.edu/~fisher/public/software/FLCT/C_VERSIONS/flct_1.06/doc/flct.pdf>`__.

        * FLCT is unable to find flows that are normal to image gradients. This
          is a defect of the LCT concept.
        * FLCT cannot determine velocities on scales below the scale size of
          structures in the images. This is a defect of the LCT concept.
        * Images that have minimal structure can give nonsensical velocity
          results.
        * Results can depend on value of ``sigma``, so you must experiment to determine
          the best choice of ``sigma``.
        * Velocities corresponding to shifts less than 0.1 - 0.2 pixels are not
          always detected. It may be necessary to increase the amount of time
          between images, depending on the noise level in the images. Sometimes
          using the filtering option helps.
        * Velocities computed within ``sigma`` pixels of the image edges can be unreliable.
        * Noisy images can result in spurious velocity results unless a suitable
          threshold value ``thresh`` is chosen.

    Parameters
    ----------
    image1 : `numpy.ndarray`
        The first image.
    image2 : `numpy.ndarray`
        The second image taken after ``deltat`` time of the first one.
    deltat : `float`
        The time interval between the two images in seconds.
    deltas : `float`
        Units of length of the side of a single pixel. Velocity is computed in units of ``deltas``/``deltat``.
    sigma : `float`
        Sub-images are weighted by Gaussian of width `sigma`. Results can depend on value of `sigma`.
        The user must experiment to determine best choice of `sigma`. If `sigma` is set to 0, only
        single values of shifts are returned. These values correspond to the overall shifts between the two images.
    order : {"row" | "column"}
        The order in which the array elements are stored that is whether they are stored as row
        major or column major.
        Defaults to `row`.
    quiet : `bool`, optional
        If set to `True` all the error messages of FLCT C code will be suppressed.
        Defaults to `False`.
    biascor : `bool`, optional
        If set to `True` bias correction will be applied while computing the velocities.
        This bias is intrinsic to the FLCT algorithm and can underestimate the velocities
        during calculations. For more
        details visit `here <http://cgem.ssl.berkeley.edu/cgi-bin/cgem/FLCT/artifact/ac3d8244c3221e8b>`__.
    thresh : `float`, optional
        A calculation will not be done for a pixel if the average absolute value
        between the two images is less than ``thresh``.
        If ``thresh`` is between 0 and 1, ``thresh`` is assumed given in
        in relative units of the maximum absolute pixel value in the average of the two images.
        Defaults to 0.
    absflag : `bool`, optional
        This is set to `True` to force the ``thresh`` values between 0 and 1 to be considered in
        absolute terms.
        Defaults to False.
    skip : `int`, optional
        The number of pixels to be skipped in the ``x`` and ``y`` direction after each calculation of a
        velocity for a pixel.
        Defaults to `None`.
    xoff : `int`, optional
        The offset in "x" direction after ``skip`` is enabled.
        Defaults to 0.
    yoff : `int`, optional
        The offset in "y" direction after ``skip`` is enabled.
        Defaults to 0.
    interp : `bool`, optional
        If set to `True` interpolation will be performed at the skipped pixels.
        Defaults to `False`.
    kr : `float`, optional
        Apply a low-pass filter to the sub-images, with a Gaussian of a characteristic wavenumber
        that is a factor of ``kr`` times the largest possible wave numbers in "x", "y" directions.
        ``kr`` should be positive.
        Defaults to `None`
    pc : `bool`, optional
        Set to `True` if the images are Plate Carrée projected.
        Defaults to `False`.
    latmin : `float`, optional
        Lower latitude limit in radians, used with ``pc``.
        Defaults to 0.
    latmax : `float`, optional
        Upper latitude limit in radians, used with ``pc``.
        Defaults to 0.2.

    Returns
    -------
    `tuple`
        A tuple containing the velocity `~numpy.ndarray` in the following order ``vx``, ``vy``, and ``vm``.
        ``vx`` is the velocity at every pixel location in the ``x`` direction.
        ``vy`` is the velocity at every pixel location in the ``y`` direction.
        ``vm`` is the mask array which is set to 1 at pixel locations where the FLCT calculations
        were performed, 0 where the calculations were not performed and 0.5 where the results were
        interpolated.

    References
    ----------
    * `FLCT C Code <http://cgem.ssl.berkeley.edu/cgi-bin/cgem/FLCT/dir?ci=tip>__`
    * `FLCT C Code Old Version <http://solarmuri.ssl.berkeley.edu/~fisher/public/software/FLCT/C_VERSIONS/>`__.
    """

    # Checking whether the C extension is correctly built.
    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    if order.lower() not in ["row", "column"]:
        raise ValueError(
            "The order of the arrays is not correctly specified. It can only be 'row' or 'column'"
        )

    # If order is column then order swap is performed.
    if order is "column":
        image1, image2 = column_row_of_two(image1, image2)
        image1 = np.array(image1)
        image2 = np.array(image2)

    if quiet is True:
        verbose = 0
    else:
        verbose = 1

    if biascor is False:
        biascor = 0
    else:
        biascor = 1

    if absflag is False:
        absflag = 0
    else:
        absflag = 1

    if interp is False:
        interp = 0
    else:
        interp = 1

    if skip is not None:
        if skip <= 0:
            raise ValueError("Skip value must be greater than zero.")

        if np.abs(xoff) >= skip or np.abs(yoff) >= skip:
            raise ValueError("The absolute value of 'xoff' and 'yoff' must be less than skip.")
    else:
        skip = 0

    if kr is not None:
        if kr <= 0.0 or kr >= 20.0:
            raise ValueError("The value of 'kr' must be between 0 and 20.")
        filter = 1
    else:
        kr = 0.0
        filter = 0

    if xoff < 0:
        xoff = skip - np.abs(xoff)
    if yoff < 0:
        yoff = skip - np.abs(yoff)

    nx = image1.shape[0]
    ny = image2.shape[1]

    if sigma == 0:
        nx = 1
        ny = 1

    if skip is not None:
        if skip >= nx or skip >= ny:
            raise ValueError("Skip is greater than the input dimensions")

    # This takes care of the order transformations in the C code.
    transp = 1

    vx = np.zeros((nx * ny,), dtype=np.float64)
    vy = np.zeros((nx * ny,), dtype=np.float64)
    vm = np.zeros((nx * ny,), dtype=np.float64)

    if pc is True:
        ierflct, vx_c, vy_c, vm_c = _pyflct.pyflct_plate_carree(
            transp,
            image1,
            image2,
            nx,
            ny,
            deltat,
            deltas,
            sigma,
            vx,
            vy,
            vm,
            thresh,
            absflag,
            filter,
            kr,
            skip,
            xoff,
            yoff,
            interp,
            latmin,
            latmax,
            biascor,
            verbose,
        )
    else:
        ierflct, vx_c, vy_c, vm_c = _pyflct.pyflct(
            transp,
            image1,
            image2,
            nx,
            ny,
            deltat,
            deltas,
            sigma,
            vx,
            vy,
            vm,
            thresh,
            absflag,
            filter,
            kr,
            skip,
            xoff,
            yoff,
            interp,
            biascor,
            verbose,
        )

    # The arrays returned from the FLCT C function are actually 2D arrays but stored as
    # single dimension array. So after getting them we need to reshape them in the original
    # shape of the input images.
    vx_c = vx_c.reshape((nx, ny))
    vy_c = vy_c.reshape((nx, ny))
    vm_c = vm_c.reshape((nx, ny))

    return (vx_c, vy_c, vm_c)
