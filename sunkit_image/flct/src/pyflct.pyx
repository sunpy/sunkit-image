# Licensed under GNU Lesser General Public License, version 2.1 - see licenses/LICENSE_FLCT.rst
# cython: language_level=3
import os
import numpy as np
cimport numpy as np
from cython cimport view

cdef extern from "./flctsubs.h":
    int flct (int transp, double * f1, double * f2, int nx, int ny, double deltat,
        double deltas, double sigma, double * vx, double * vy, double * vm,
        double thresh, int absflag, int filter, double kr, int skip,
        int poffset, int qoffset, int interpolate, int biascor, int verbose)
    int read2images (char *fname, int * nx, int * ny, double **arr, double **barr,
            int transp)
    int write3images (char *fname, double *arr, double *barr, double *carr,
        int nx, int ny, int transp)
    int is_large_endian()
    int flct_pc (int transp, double * f1, double * f2, int nx, int ny, double deltat,
        double deltas, double sigma, double * vx, double * vy, double * vm,
        double thresh, int absflag, int filter, double kr, int skip,
        int poffset, int qoffset, int interpolate, double latmin, double latmax,
        int biascor, int verbose)
    int write2images (char *fname, double *arr, double *barr, int nx, int ny, int transp)

cdef extern from "./sunkit.h":
    int read3images (char *fname, int * nx, int * ny, double **arr, double **barr, double **carr, int transp)

np.import_array()
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

def read_two_images(file_name, transpose=0):

    cdef int nx
    cdef int ny
    cdef double *arr
    cdef double *barr
    file_name = file_name.encode('utf-8')

    ier = read2images(file_name, &nx, &ny, &arr, &barr, transpose)

    cdef view.array cy_arr = <double[:nx, :ny]> arr
    cdef view.array cy_barr = <double[:nx, :ny]> barr

    a = np.array(cy_arr)
    b = np.array(cy_barr)

    return (ier, a, b)

def read_three_images(file_name, transpose=0):

    cdef int nx
    cdef int ny
    cdef double *arr
    cdef double *barr
    cdef double *carr
    file_name = file_name.encode('utf-8')

    ier = read3images(file_name, &nx, &ny, &arr, &barr, &carr,transpose)

    cdef view.array cy_arr = <double[:nx, :ny]> arr
    cdef view.array cy_barr = <double[:nx, :ny]> barr
    cdef view.array cy_carr = <double[:nx, :ny]> carr

    a = np.array(cy_arr)
    b = np.array(cy_barr)
    c = np.array(cy_carr)

    return (ier, a, b, c)


def write_two_images(file_name, arr, barr, transpose=0):

    nx, ny = arr.shape

    arr = arr.reshape((-1))
    barr = barr.reshape((-1))

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] arr_c = np.ascontiguousarray(arr, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] barr_c = np.ascontiguousarray(barr, dtype = np.double)

    file_name = file_name.encode('utf-8')
    ier = write2images(file_name, <double *> arr_c.data, <double *> barr_c.data, nx, ny, transpose)

    return ier


def write_three_images(file_name, arr, barr, carr, transpose=0):

    nx, ny = arr.shape

    arr = arr.reshape((-1))
    barr = barr.reshape((-1))
    carr = carr.reshape((-1))

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] arr_c = np.ascontiguousarray(arr, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] barr_c = np.ascontiguousarray(barr, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] carr_c = np.ascontiguousarray(carr, dtype = np.double)

    file_name = file_name.encode('utf-8')
    ier = write3images(file_name, <double *> arr_c.data, <double *> barr_c.data, <double *> carr_c.data, nx, ny, transpose)

    return ier

def pyflct_plate_carree(transpose, f1, f2, nxorig, nyorig, deltat, deltas, sigma,
                      vx, vy, vm, thresh, absflag, filter, kr, skip, poffset,
                      qoffset, interpolate, latmin, latmax, biascor, verbose
):
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] f1_c = np.ascontiguousarray(f1, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] f2_c = np.ascontiguousarray(f2, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] vx_c = np.ascontiguousarray(vx, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] vy_c = np.ascontiguousarray(vy, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] vm_c = np.ascontiguousarray(vm, dtype = np.double)

    ierflct = flct_pc(transpose, <double *> f1_c.data, <double *> f2_c.data, nxorig, nyorig, deltat,
                      deltas, sigma, <double *> vx_c.data, <double *> vy_c.data, <double *> vm_c.data,
                      thresh, absflag, filter, kr, skip, poffset, qoffset, interpolate, latmin, latmax,
                      biascor, verbose)

    return ierflct, vx_c, vy_c, vm_c


def pyflct(transpose, f1, f2, nxorig, nyorig, deltat, deltas, sigma,
           vx, vy, vm, thresh, absflag, filter, kr, skip,
           poffset, qoffset, interpolate, biascor, verbose
):
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] f1_c = np.ascontiguousarray(f1, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=2, mode="c"] f2_c = np.ascontiguousarray(f2, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] vx_c = np.ascontiguousarray(vx, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] vy_c = np.ascontiguousarray(vy, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] vm_c = np.ascontiguousarray(vm, dtype = np.double)

    ierflct = flct(transpose, <double *> f1_c.data, <double *> f2_c.data, nxorig, nyorig, deltat,
                   deltas, sigma, <double *> vx_c.data, <double *> vy_c.data, <double *> vm_c.data,
                   thresh, absflag, filter, kr, skip, poffset, qoffset, interpolate, biascor, verbose)

    return ierflct, vx_c, vy_c, vm_c

# This is created to deal with the arrays which were first read by IDL
def swap_order_two(arr, barr):

    ier = write_two_images("temp.dat", arr, barr, 0)

    ier, cy_arr, cy_barr = read_two_images("temp.dat", transpose=1)
    os.remove("temp.dat")

    a = np.array(cy_arr)
    b = np.array(cy_barr)

    return a, b


def swap_order_three(arr, barr, carr):

    temp = np.zeros_like(arr)

    arr, barr = swap_order_two(arr,barr)
    carr, temp = swap_order_two(carr, temp)

    del temp

    return arr, barr, carr
