import numpy as np
import math
import struct

from sunkit_image.flct import pyflctsubs

__all__ = ["flct", ]


def vcimageout(data, filename='./input.dat'):
    '''
    Adapted from vcimage1/2/3out.pro in Fisher & Welsch 2008.
    Input: data - tuple containing all image data to be stored in filename
                  It must be in the following format:
                  data = (data1, data2 ...)
                  where, data1, data2 ... are image data at different times
           filename - name of file that will be generated
    Output: Return - None
            A new file with name of filename will be generated containing
            data.
    '''

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
            print('vcimage2out: dimensions or ranks of data ' +
                  'do not match')
            return None
    for v in shapes:
        if len(v) != 2 and sizes[0] > 1:
            print('vcimage2out: input array is not 2d')
            return None
    # initial integer ID for a "vel_ccor" i/o file
    vcid = int(2136967593).to_bytes(4, 'big')
    if sizes[0] > 1:
        nx = int(data[0].shape[0])
        ny = int(data[0].shape[1])
    else:
        nx = int(1)
        ny = int(1)
    nx = nx.to_bytes(4, 'big')
    ny = ny.to_bytes(4, 'big')
    f = open(filename, 'wb')
    f.write(vcid)
    f.write(nx)
    f.write(ny)
    for i in range(num):
        array = data[i]
        for value in np.nditer(array, order='F'):
            v = struct.pack('>f', value)
            f.write(v)

    f.close()


def vcimagein(filename='output.dat'):
    '''
    Adapted from vcimage1/2/3out.pro in Fisher & Welsch 2008.
    Input: filename - name of C binary file
    Output: Return - a list containing arrays of data stored in that file.
    '''
    f = open(filename, 'rb')
    vcid = struct.unpack('>i', f.read(4))[0]
    if vcid != 2136967593:
        print('Input file is not a vel_coor i/o file')
        f.close()
        return None
    nx = struct.unpack('>i', f.read(4))[0]
    ny = struct.unpack('>i', f.read(4))[0]
    data = f.read()
    f.close()
    # calculate number of files
    num = int(len(data) / (4. * nx * ny))
    f.close()
    array = ()
    for k in range(num):
        dust = np.zeros((nx, ny), dtype=np.float32)
        offset = nx * ny * k * 4
        idx = offset
        # In the case when sigma is set to zero using FLCT
        if nx == 1 and ny == 1:
            v = struct.unpack('>f', data[idx:idx+4])[0]
            array = array + (v,)
        else:
            it = np.nditer(dust, flags=["multi_index"],
                           op_flags=["readwrite"], order='F')
            while not it.finished:
                v = struct.unpack('>f', data[idx:idx+4])[0]
                dust[it.multi_index] = v
                it.iternext()
                idx = idx + 4
            array = array + (dust,)

    return array


def flct(image1, image2, deltat, deltas, sigma, quiet=False,
         biascor=False, thresh=0., absflag=False, skip=None, poff=0, qoff=0,
         skipon=0, interp=False, kr=None, pc=False, latmin=0, latmax=0.2,
         ):

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

    if kr is not None:
        if kr <= 0. or kr >= 20.:
            raise ValueError("The value of kr must be between 0. and 20.")
        filter = 1
    else:
        kr = 0.
        filter = 0

    ibe = pyflctsubs.endian()

    # The below statements are not needed since we are taking numpy arrays as imput
    # vcimageout((image1, image2))
    # infile = b'input.dat'

    # ier, nx, ny, f1, f2 = pyflctsubs.read_to_images(infile, 0)

    # image1 = np.array(f1)
    # image2 = np.array(f2)

    # nxorig = nx
    # nyorig = ny

    nx = image1.shape[0]
    ny = image2.shape[1]

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
        ierflct, vx_c, vy_c, vm_c = pyflctsubs.pyflct_plate_carree(transp, image1, image2, nx, ny,
                                                                   deltat, deltas, sigma, vx, vy,
                                                                   vm, thresh, absflag, filter, kr,
                                                                   skip, poff, qoff, interp, latmin,
                                                                   latmax, biascor, verbose)
    else:
        ierflct, vx_c, vy_c, vm_c = pyflctsubs.pyflct(transp, image1, image2, nx, ny, deltat,
                                                      deltas, sigma, vx, vy, vm, thresh, absflag,
                                                      filter, kr, skip, poff, qoff, interp, biascor,
                                                      verbose)

    # This is also not needed as numpy arrays are returned
    # outfile = b'output.dat'
    # pyflctsubs.write_3_images(outfile, vx_c, vy_c, vm_c, nx, ny, transp)

    return (vx_c, vy_c, vm_c)
