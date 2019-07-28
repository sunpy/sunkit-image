import math
import struct

import numpy as np

try:
    from sunkit_image.flct import _pyflct
except ImportError:
    _pyflct = None

__all__ = ["flct",]


def flct(
    image1,
    image2,
    order,
    deltat,
    deltas,
    sigma,
    quiet=False,
    biascor=False,
    thresh=0.0,
    absflag=False,
    skip=None,
    poff=0,
    qoff=0,
    interp=False,
    kr=None,
    pc=False,
    latmin=0,
    latmax=0.2,
):
    """
    A python wrapper which calls the FLCT C routines to perform Fourier Linear Correlation
    Tracking between two images taken at some interval of time.

    .. note::

        * In the references there are some dat files which can be used to test the FLCT code. The
          best method to read those dat files is the `_pyflct.read_two_images` and `_pyflct.read_three_images`
          as the arrays would automatically be read in row major format.
        * If you use the IDL IO routines to get the input arrays from dat files as mentioned on the
          FLCT README given in the references, the IDL routines always read the binary files in the
          column major, but both python and C, on which these functions are based row major so the
          order of the arrays have to be changed and read from the binary files in row major order.
          This may lead to different values in both the cases.
        * The above has already been taken care of in this module. If your input arrays are column
          major then pass the `order` parameter as `column` and it will automatically take care of
          the order change and values. But this can produce some changes in the values of the arrays
        * If you have arrays in row major then you can pass `order` parameter as `row` and no order
          change will be performed.

    Parameters
    ----------
    image1 : `numpy.ndarray`
        The first image of the sequence of two images on which the procedure is to be perfromed.
    image2 : `numpy.ndarray`
        The second image of the sequence of two images taken after `deltat` time of the first one.
    order : `string`
        The order in which the array elements are stored that is whether they are stored as row
        major or column major.
    deltat : `float`
        The time interval between the capture of the two images.
    deltas : `float`
        Units of length of the side of a single pixel
    sigma : `float`
        The width of Gaussian kernel with which the images are to be modulated. If sigma is `0` then
        the overall shift between the images is returned.
    quiet : `bool`
        If set to `True` all the error messages of FLCT C code will be supressed.
        Defaults to `False`.
    biascor : `bool`
        If set to `True` bias correction will be applied while computing the velocities.
    thresh : `float`
        The threshold value below which if the average absolute value of pixel values for a certain
        pixel in both the images, falls the FLCT calculation will not be done for that pixel.  If
        thresh is between 0 and 1, thresh is assumed given in units relative to the largest
        absolute value of the image averages.
        Defaults to 0.
    absflag : `bool`
        This is set to `True` to force the `thresh` values between 0 and 1 to be considered in the
        absolute terms.
        Defaults to False.
    skip : `int`
        The number of pixels to be skipped in the x and y direction after each calculation of a
        velocity for a pixel.
        Defaults to `None`.
    poff : `int`
        The offset in x direction after skip is enabled.
        Defaults to 0.
    qoff : `int`
        The offset in y direction after skip is enabled.
        Defaults to 0.
    interp : `bool`
        If set to `True` interpolation will be performed at the skipped pixels.
        Defaults to `False`.
    kr : `float`
        Filter sub-images.
        Defaults to `None`
    pc : `bool`
        Set to `True` if the images are in Plate Carree.
        Defaults to `False`.
    latmin : `float`
        Lower latitude limit in radians.
        Defaults to 0.
    latmax : `float`
        Upper latitude limit in radians.
        Defaults to 0.2

    Returns
    -------
    `tuple`
        A tuple containing the velocity arrays in the following order vx, vy, and vm.

    References
    ----------
    * The FLCT software package which can be found here:
      http://solarmuri.ssl.berkeley.edu/~fisher/public/software/FLCT/C_VERSIONS/
    """

    # Checking whether the C extension is correctly built.
    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    if order != "row" and order != "column":
        raise ValueError("The order of the arrays is not correctly specifed. It can only be 'row' or 'column'")

    # If order is column then order swap is performed.
    if order is "column":
        image1, image2 = _pyflct.swap_order_two(image1, image2)
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
        skipon = skip + math.abs(qoff) + math.abs(poff)

        if math.abs(poff) >= skip or math.abs(qoff) >= skip:
            raise ValueError("The absolute value of poff and qoff must be less than skip")
    else:
        skip = 0
        skipon = 0

    if kr is not None:
        if kr <= 0.0 or kr >= 20.0:
            raise ValueError("The value of kr must be between 0. and 20.")
        filter = 1
    else:
        kr = 0.0
        filter = 0

    if(poff < 0):
        poff = skip - math.abs(poff)
    if(qoff < 0):
        qoff = skip - math.abs(qoff)

    nx = image1.shape[0]
    ny = image2.shape[1]

    nxorig = nx
    nyorig = ny

    if sigma == 0:
        nx = 1
        ny = 1

    if skip is not None:
        if skip >= nx or skip >= ny:
            raise ValueError("Skip is greater than the input dimensions")

    transp = 1

    vx = np.zeros((nx * ny,), dtype=float)
    vy = np.zeros((nx * ny,), dtype=float)
    vm = np.zeros((nx * ny,), dtype=float)

    if pc is True:
        ierflct, vx_c, vy_c, vm_c = _pyflct.pyflct_plate_carree(
            transp,
            image1,
            image2,
            nxorig,
            nyorig,
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
            poff,
            qoff,
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
            nxorig,
            nyorig,
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
            poff,
            qoff,
            interp,
            biascor,
            verbose,
        )

    vx_c = vx_c.reshape((nxorig, nyorig))
    vy_c = vy_c.reshape((nxorig, nyorig))
    vm_c = vm_c.reshape((nxorig, nyorig))

    return (vx_c, vy_c, vm_c)
