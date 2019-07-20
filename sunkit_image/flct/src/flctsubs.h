/*

 FLCT: http://solarmuri.ssl.berkeley.edu/overview/publicdownloads/software.html
 Copyright (C) 2007-2018 Regents of the University of California

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; either version 2 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program; if not, write to the Free Software
 Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

*/

# include <stdio.h>
# include <string.h>
# include <ctype.h>
# include <stdlib.h>
# include <math.h>

/* To include C99 complex arithmetic, uncomment the following line defining
   COMPLEXH.  To not include C99 complex arithmetic, leave this definition
   commented out. */

/* # define COMPLEXH 1 */

# ifdef COMPLEXH
   # include <complex.h>   
#endif

# include <fftw3.h> 

/* To write files deriv2.dat and deriv1.dat, containing 2nd derivatives of
the cross-correlation function, and the peak value and first derivatives,
uncomment the line below defining CCDATA: */

/* # define CCDATA 1 */

/* global declarations */

/* i4 and f4 are supposed to be definitions that give rise to 4 byte integers
 * and 4 byte floats */


typedef int i4;
typedef float f4;

/* function prototypes: */

/* flct function calling arguments not yet completely defined */
void flct_f77__(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny,
    double * deltat, double * deltas, double * sigma, double * vx,
    double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter,
    double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
    i4 * biascor, i4 * verbose);
void flct_f77_(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny,
    double * deltat, double * deltas, double * sigma, double * vx,
    double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter,
    double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
    i4 * biascor, i4 * verbose);
void flct_f77(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny,
    double * deltat, double * deltas, double * sigma, double * vx,
    double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter,
    double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
    i4 * biascor, i4 * verbose);
i4 flct (i4 transp, double * f1, double * f2, i4 nx, i4 ny, double deltat,
    double deltas, double sigma, double * vx, double * vy, double * vm,
    double thresh, i4 absflag, i4 filter, double kr, i4 skip,
    i4 poffset, i4 qoffset, i4 interpolate, i4 biascor, i4 verbose);
i4 readimage (char *fname, i4 *nx, i4 * ny, double **arr, i4 transp);
i4 read2images (char *fname, i4 * nx, i4 * ny, double **arr, double **barr,
        i4 transp);
i4 where (char *cond, i4 xsize, i4 ** index, i4 * length_index);
i4 cross_cor (i4 init, i4 hires, i4 expand, double *arr, double *barr,
    double **absccor, i4 nx, i4 ny, double *shiftx, double *shifty, 
        i4 filterflag, double kr, double sigma);
i4 writeimage (char *fname, double *arr, i4 nx, i4 ny, i4 transp);
i4 write2images (char *fname, double *arr, double *barr, i4 nx, i4 ny,
        i4 transp);
i4 write3images (char *fname, double *arr, double *barr, double *carr,
        i4 nx, i4 ny, i4 transp);
i4 shift2d (double *arr, i4 nx, i4 ny, i4 ishift, i4 jshift);
i4 maxloc (double *arr, i4 xsize);
i4 minloc (double *arr, i4 xsize);
i4 iminloc (i4 * arr, i4 xsize);
i4 imaxloc (i4 * arr, i4 xsize);
double r (double t);
i4 interpcc2d (double *fdata, double xmiss, i4 nx, i4 ny, 
double *xwant, i4 nxinterp, double *ywant, i4 nyinterp, double **finterp);
i4 gaussfilt(double *filter, double *kx, double *ky, i4 nx, i4 ny, double kr);
i4 make_freq(double *k, i4 ndim);
i4 filter_image(double *arr, double *barr, double *outarr, double *outbarr,
        i4 nx, i4 ny, double kr);
i4 is_large_endian (void);
i4 byteswapflct (unsigned char *arr, i4 arrsize, i4 nbpw);
i4 warp_frac2d(double *arr, double *delx, double *dely, double *outarr,
        i4 nx, i4 ny, i4 transp, i4 verbose);
i4 flct_pc (i4 transp, double * f1, double * f2, i4 nx, i4 ny, double deltat,
    double deltas, double sigma, double * vx, double * vy, double * vm,
    double thresh, i4 absflag, i4 filter, double kr, i4 skip,
    i4 poffset, i4 qoffset, i4 interpolate, double latmin, double latmax,
    i4 biascor, i4 verbose);
void flct_pc_f77__(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny,
    double * deltat, double * deltas, double * sigma, double * vx,
    double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter,
    double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
    double * latmin, double * latmax, i4 * biascor, i4 * verbose);
void flct_pc_f77_(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny,
    double * deltat, double * deltas, double * sigma, double * vx,
    double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter,
    double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
    double * latmin, double * latmax, i4 * biascor, i4 * verbose);
void flct_pc_f77(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny,
    double * deltat, double * deltas, double * sigma, double * vx,
    double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter,
    double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
    double * latmin, double * latmax, i4 * biascor, i4 * verbose);
i4 mc2pc(i4 transp, double *f, i4 nxinterp, i4 nyinterp, double umin,
        double umax, double vmin, double vmax, double ** finterp, i4 nx,
        i4 ny);
i4 pc2mc(i4 transp, double *f, i4 nx, i4 ny, double lonmin, double lonmax,
        double latmin, double latmax, double ** finterp, i4 *nxinterp,
        i4 *nyinterp, double * vmin, double *vmax);
double signum(double v);
i4 shift_frac2d(double *arr, double delx, double dely, double *outarr,
        i4 nx, i4 ny, i4 transp, i4 verbose);
void warp_frac2d_f77(double *arr, double *delx, double *dely, double *outarr,
    i4 *nx, i4*ny, i4 *transp, i4 *verbose);
void warp_frac2d_f77_(double *arr, double *delx, double *dely, double *outarr,
    i4 *nx, i4*ny, i4 *transp, i4 *verbose);
void warp_frac2d_f77__(double *arr, double *delx, double *dely, double *outarr,
    i4 *nx, i4*ny, i4 *transp, i4 *verbose);
void shift_frac2d_f77(double *arr, double *delx, double *dely, double *outarr,
    i4 *nx, i4 *ny, i4 *transp, i4 *verbose);
void shift_frac2d_f77_(double *arr, double *delx, double *dely, double *outarr,
    i4 *nx, i4 *ny, i4 *transp, i4 *verbose);
void shift_frac2d_f77__(double *arr, double *delx, double *dely, double *outarr,
    i4 *nx, i4 *ny, i4 *transp, i4 *verbose);

/* end function prototypes */
