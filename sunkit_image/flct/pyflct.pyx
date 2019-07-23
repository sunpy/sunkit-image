# cython: language_level=3
import numpy as np
cimport numpy as np
from cython cimport view

cdef extern from "./src/flctsubs.h":
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

np.import_array()
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

def read_to_images(file_name, transpose=0):

    cdef int nx
    cdef int ny
    cdef double *arr
    cdef double *barr

    ier = read2images(file_name, &nx, &ny, &arr, &barr, transpose)

    cdef view.array cy_arr = <double[:nx, :ny]> arr
    cdef view.array cy_barr = <double[:nx, :ny]> barr

    return (ier, nx, ny, cy_arr, cy_barr)


def write_3_images(file_name, arr, barr, carr, nx, ny, transpose):
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] arr_c = np.ascontiguousarray(arr, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] barr_c = np.ascontiguousarray(barr, dtype = np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] carr_c = np.ascontiguousarray(carr, dtype = np.double)

    ier = write3images(file_name, <double *> arr_c.data, <double *> barr_c.data, <double *> carr_c.data, nx, ny, transpose)


def endian():
    return is_large_endian ()


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
