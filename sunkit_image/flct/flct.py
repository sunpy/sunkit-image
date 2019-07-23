import math
import struct

import numpy as np

try:
    from sunkit_image.flct import _pyflct
except ImportError:
    _pyflct = None

__all__ = ["flct", "vcimageout", "vcimagein"]


def vcimageout(data, filename="./input.dat"):
    """

    """

    # Perform size and shape check
    num = len(data)
    shapes = []
    sizes = []
    for i in range(num):
        array = data[i]
        array = np.array(array, dtype=np.float32)
        shapes.append(array.shape)
        sizes.append(array.size)
    for v in sizes:
        if v != sizes[0]:
            print("vcimage2out: dimensions or ranks of data " + "do not match")
            return None
    for v in shapes:
        if len(v) != 2 and sizes[0] > 1:
            print("vcimage2out: input array is not 2d")
            return None
    # initial integer ID for a "vel_ccor" i/o file
    vcid = int(2136967593).to_bytes(4, "big")
    if sizes[0] > 1:
        nx = int(data[0].shape[0])
        ny = int(data[0].shape[1])
    else:
        nx = int(1)
        ny = int(1)
    nx = nx.to_bytes(4, "big")
    ny = ny.to_bytes(4, "big")
    f = open(filename, "wb")
    f.write(vcid)
    f.write(nx)
    f.write(ny)
    for i in range(num):
        array = data[i]
        for value in np.nditer(array, order="F"):
            v = struct.pack(">f", value)
            f.write(v)

    f.close()


def vcimagein(filename="output.dat"):
    """
    Adapted from vcimage1/2/3out.pro in Fisher & Welsch 2008.
    Input: filename - name of C binary file
    Output: Return - a list containing arrays of data stored in that file.
    """
    f = open(filename, "rb")
    vcid = struct.unpack(">i", f.read(4))[0]
    if vcid != 2136967593:
        print("Input file is not a vel_coor i/o file")
        f.close()
        return None
    nx = struct.unpack(">i", f.read(4))[0]
    ny = struct.unpack(">i", f.read(4))[0]
    data = f.read()
    f.close()
    # calculate number of files
    num = int(len(data) / (4.0 * nx * ny))
    f.close()
    array = ()
    for k in range(num):
        dust = np.zeros((nx, ny), dtype=np.float32)
        offset = nx * ny * k * 4
        idx = offset
        # In the case when sigma is set to zero using FLCT
        if nx == 1 and ny == 1:
            v = struct.unpack(">f", data[idx : idx + 4])[0]
            array = array + (v,)
        else:
            it = np.nditer(dust, flags=["multi_index"], op_flags=["readwrite"], order="F")
            while not it.finished:
                v = struct.unpack(">f", data[idx : idx + 4])[0]
                dust[it.multi_index] = v
                it.iternext()
                idx = idx + 4
            array = array + (dust,)

    return array


def flct(
    image1,
    image2,
    deltat,
    deltas,
    sigma,
    quiet=False,
    biascor=False,
    thresh=0.0,
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

    Parameters
    ----------
    image1 : `numpy.ndarray`
        The first image of the sequence of two images on which the procedure is to be perfromed.
    image2 : `numpy.ndarray`
        The second image of the sequence of two images taken after `deltat` time of the first one.
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
        pixel in both the images, falls the FLCT calculation will not be done for that pixel.
        Defaults to 0.
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

    if _pyflct is None:
        raise ImportError("C extension for flct is missing, please rebuild.")

    if quiet is True:
        verbose = 0
    else:
        verbose = 1

    if biascor is False:
        biascor = 0
    else:
        biascor = 1

    if thresh != 0.0:
        absflag = 1
    else:
        absflag = 0

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

    # ibe = pyflctsubs.endian()

    # The below statements are not needed since we are taking numpy arrays as imput
    # vcimageout((image1, image2))
    # infile = b'input.dat'

    # ier, nx, ny, f1, f2 = pyflctsubs.read_to_images(infile, 0)

    # image1 = np.array(f1)
    # image2 = np.array(f2)

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

    print("All inputs")
    print(transp)
    print(image1)
    print(image2)
    print(
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

    # This is also not needed as numpy arrays are returned
    # outfile = b'output.dat'
    # pyflctsubs.write_3_images(outfile, vx_c, vy_c, vm_c, nx, ny, transp)

    vx_c = vx_c.reshape((nxorig, nyorig))
    vy_c = vy_c.reshape((nxorig, nyorig))
    vm_c = vm_c.reshape((nxorig, nyorig))

    return (vx_c, vy_c, vm_c)
