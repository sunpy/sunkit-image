/*
       FLCT Local Correlation Tracking software
       http://cgem.ssl.berkeley.edu/~fisher/public/software/FLCT
       Copyright (C) 2007-2019, Regents of the University of California

       This software is based on the concepts described in Welsch & Fisher
       (2008, PASP Conf. Series 383, 373), with updates described in
       Fisher et al. 2019, "The PDFI_SS Electric Field Inversion Software",
       in prep.
       If you use the software in a scientific
       publication, the authors would appreciate a citation to these papers
       and any future papers describing updates to the methods.

       This is free software; you can redistribute it and/or
       modify it under the terms of the GNU Lesser General Public
       License version 2.1 as published by the Free Software Foundation.

       This software is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
       See the GNU Lesser General Public License for more details.

       To view the GNU Lesser General Public License visit
       http://www.gnu.org/copyleft/lesser.html
       or write to the Free Software Foundation, Inc.,
       59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
# include "sunkit.h"

i4 read3images (char *fname, i4 * nx, i4 * ny, double **arr, double **barr,
	     double **carr, i4 transp)
/* Function to read array dims, create space for 2 double arrays, read them in.
 * Note the double pointer to the double precision * arrays in the calling
 * argument.  Note also these are referenced as pointers and returned to the
 * calling function. */
{
  FILE *f1;			/* pointer to input file */
  i4 newsize;			/* size of the new double prec. array to be
				   read in =nx*ny */
  i4 i, ier, ibe, ise, vcid;
  f4 *farr, *fbarr, *fcarr;
  ibe = is_large_endian ();
  ise = 0;
  if (ibe == 0) ise = 1;	/* set small endian flag if not big  endian */
  f1 = fopen (fname, "rb");	/* open file for binary unformatted read */
  ier = 0;
  if (f1 == NULL)		/* error exit if file can't be opened */
    {
      printf ("read3images: cannot open file %s\n", fname);
      exit (1);
    }

  /* check that files begins with special vel_ccor flag: */
  fread (&vcid, sizeof (i4), 1, f1);
  if (ise) byteswapflct ((void *) &vcid, 1, sizeof (i4));
  if (vcid != 2136967593)
    {
      printf ("read3images: input file is not a vel_ccor i/o file\n");
      exit (1);
    }

  /* order of nx, ny read in must be switched if transp is true */

  if (transp)
    {
      fread (ny, sizeof (i4), 1, f1);
      fread (nx, sizeof (i4), 1, f1);
    }
  else
    {
      fread (nx, sizeof (i4), 1, f1);
      fread (ny, sizeof (i4), 1, f1);
    }
  if (ise)			/* byteswap nx,ny if small endian */
    {
      byteswapflct ((void *) nx, 1, sizeof (i4));
      byteswapflct ((void *) ny, 1, sizeof (i4));
    }
/*
	printf("\n\nnx,ny read in from file arr = %d,%d\n",*nx,*ny);
*/
  newsize = (*nx) * (*ny) * sizeof (double);	/* size of new double array */

  /* now create enough space in memory to hold the array arr */

  *arr = malloc (newsize);

  /* allocate space for the temporary, f4 array farr */

  farr = malloc (sizeof (f4) * (*nx) * (*ny));

  if (!*arr)
    {				/* check for error in memory allocation */
      printf ("read3images: memory request for arr failed\n");
      exit (1);
    }
/*
	printf("%d bytes of memory now allocated for arr \n",newsize);
*/

  /* now read in the arr array */

  fread (farr, sizeof (f4), (*nx) * (*ny), f1);
  if (ise) byteswapflct ((void *) farr, (*nx) * (*ny), sizeof (f4));
  /*byteswap if needed */

  /* now create space for temp. f4 array fbarr, and returned
   * array barr: */

  fbarr = malloc (sizeof (f4) * (*nx) * (*ny));
  *barr = malloc (newsize);

  if (!*barr)
    {				/* check for error in memory allocation */
      printf ("read3images: memory request for barr failed\n");
      exit (1);
    }

  /* now read in the fbarr array */

  fread (fbarr, sizeof (f4), (*nx) * (*ny), f1);
  /*byteswap if needed */
  if (ise) byteswapflct ((void *) fbarr, (*nx) * (*ny), sizeof (f4));

  fcarr = malloc (sizeof (f4) * (*nx) * (*ny));
  *carr = malloc (newsize);

  if (!*carr)
    {				/* check for error in memory allocation */
      printf ("read3images: memory request for barr failed\n");
      exit (1);
    }

  /* now read in the fcarr array */

  fread (fcarr, sizeof (f4), (*nx) * (*ny), f1);
  /*byteswap if needed */
  if (ise) byteswapflct ((void *) fcarr, (*nx) * (*ny), sizeof (f4));

  /* now transfer data from temp. arrays to arr and barr: */

  for (i = 0; i < (*nx) * (*ny); i++)
    {
      *(*arr + i) = (double) *(farr + i);
      *(*barr + i) = (double) *(fbarr + i);
      *(*carr + i) = (double) *(fcarr + i);
    }

  /* now free the temp. arrays and close the files */

  free (farr);
  free (fbarr);
  free(fcarr);
  fclose (f1);
  ier = 1;
  return ier;
}
