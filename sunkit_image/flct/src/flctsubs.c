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

# include "flctsubs.h"

void flct_f77__(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny, 
     double * deltat, double * deltas, double * sigma, double * vx, 
     double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter, 
     double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
     i4 * biascor, i4 * verbose)

{
i4 ierflct;

/* Now call the C version of the flct function, dereferencing all the
   variables that aren't pointers within the C flct function */

ierflct=flct(*transp,f1,f2,*nx,*ny,*deltat,*deltas,*sigma,vx,vy,vm,
     *thresh,*absflag,*filter,*kr,*skip,*poffset,*qoffset,*interpolate,
     *biascor,*verbose);

return;
}

void flct_f77_(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny, 
     double * deltat, double * deltas, double * sigma, double * vx, 
     double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter, 
     double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
     i4 * biascor, i4 * verbose)

{
i4 ierflct;

/* Now call the C version of the flct function, dereferencing all the
   variables that aren't pointers within the C flct function */

ierflct=flct(*transp,f1,f2,*nx,*ny,*deltat,*deltas,*sigma,vx,vy,vm,
     *thresh,*absflag,*filter,*kr,*skip,*poffset,*qoffset,*interpolate,
     *biascor,*verbose);

return;
}

void flct_f77(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny, 
     double * deltat, double * deltas, double * sigma, double * vx, 
     double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter, 
     double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
     i4 * biascor, i4 * verbose)

{
i4 ierflct;

/* Now call the C version of the flct function, dereferencing all the
   variables that aren't pointers within the C flct function */

ierflct=flct(*transp,f1,f2,*nx,*ny,*deltat,*deltas,*sigma,vx,vy,vm,
     *thresh,*absflag,*filter,*kr,*skip,*poffset,*qoffset,*interpolate,
     *biascor,*verbose);

return;
}

i4 flct (i4 transp, double * f1, double * f2, i4 nx, i4 ny, double deltat, 
    double deltas, double sigma, double * vx, double * vy, double * vm,
    double thresh, i4 absflag, i4 filter, double kr, i4 skip,
    i4 poffset, i4 qoffset, i4 interpolate, i4 biascor, i4 verbose) 
{

/* BEGIN FLCT FUNCTION */

/*  char *version ="1.05    "; */

  i4 hires=-1, expand=1, sigmaeq0;
  i4 nxtmp, nytmp, nxorig, nyorig, i, j, ii, jj, ic, jc, ixcount=0, iycount=0;
  i4 icc, nsize=0, nt, ndmin, init;
  i4 iloc1, iloc2, imin0, jmax0, jmin0, imax0, imin, jmin, imax, jmax,
    isize, jsize;
  i4 skipon=0, skipxy,noskipx,noskipy,noskipxy;
  i4 xoffset=0,yoffset=0 ;
  double *vxnoi=NULL, *vynoi=NULL, *vmnoi=NULL, *xwant, *ywant, *vxint, *vyint;
  double *gaussdata, *f1temp, *f2temp,
    *g1, *g2, *absccor;
  double shiftx, shifty, argx, argy, f1max, f2max, fmax, fabsbar, vmask;
  double sigminv=-999., 
     deltinv, vxx=0., vyy=0., f1bar=0., f2bar=0.;
/*      double tol=1e-4; */
  double tol = 1e-2;
  double xmiss = -999999.;
  i4 hardworkneeded, belowthresh;
#ifdef CCDATA
  char *gamfile="deriv2.dat";
  char *peakfile="deriv1.dat";
#endif
  i4 maxind,ixmax,iymax,absccmax,nsig;
  double gamx2val,gamy2val,gamxyval,
     gamxval,gamyval, gampeakval;
  double *gamx2=NULL, *gamy2=NULL, *gamx=NULL, *gamy=NULL, *gamxy=NULL, 
     *gampeak=NULL;
  double fx,fy,fxx,fyy,fxy,fpeak,hessian,hm1over2,corfac,gam2oversigma2;

/* First, need to interchange roles of nx, ny if (transp) is true: */

  if(transp)
  {
     nxtmp=nx;
     nytmp=ny;
     nx=nytmp;
     ny=nxtmp;
  }

  /* nx, ny may get set to 1 if sigma=0. so copy original values */

  nxorig=nx;
  nyorig=ny;

/* get 1/deltat: */

  deltinv = 1. / deltat;

/* Get information out of sigma: */

  if(sigma > 0.)
  {
     sigminv = 1. / sigma;
     sigmaeq0=0;
  }
  else
  {
     sigmaeq0 = 1;
  }

  if(!sigmaeq0)
  {
     /* This stuff done only if sigma > 0 */
     nt = (i4) sigma *sqrt (log (1. / tol));
     ndmin = (nx < ny) ? (((nx / 3) / 2)) * 2 : (((ny / 3) / 2)) * 2;
     nsize = ((nt / 2) * 2 < ndmin) ? (nt / 2) * 2 : ndmin;
     if (verbose) 
     {
       printf ("flct: nominal sliding box size = %d\n", 2 * nsize);
       fflush(stdout);
     }
     if(nsize <= 0)
     {
        printf("flct: error - illegal box size, exiting\n");
        fflush(stdout);
        exit(1);
     }
  }
  if(sigmaeq0) 
  {
     /* sigma = 0 means we'll only compute one point */
     nx=1;
     ny=1;
  }

  /* figure out if threshold is in absolute or fractional units
   * and if fractional, convert to absolute units.  If thresh is between
   * zero and 1 (not inclusive) then it's assumed to be fractional,
   * (unless the threshold string ends with 'a') and must be scaled.
   * if aloc == NULL, there's no 'a' in the threshold string. */

  if(!sigmaeq0)
  {
    if ((thresh > 0.) && (thresh < 1.) && (absflag == 0))
      {
        f1temp = (double *) malloc (sizeof (double) * nx * ny);
        f2temp = (double *) malloc (sizeof (double) * nx * ny);

        for (i = 0; i < nx * ny; i++)

  	{

	  /* compute abs value of f1,f2 arrays as f1temp,
	   * f2temp arrays */

	  *(f1temp + i) = (double) fabs (*(f1 + i));
	  *(f2temp + i) = (double) fabs (*(f2 + i));
	}

      /* now find maximum absolute value of both images */

        iloc1 = maxloc (f1temp, nx * ny);
        iloc2 = maxloc (f2temp, nx * ny);
        f1max = *(f1temp + iloc1);
        f2max = *(f2temp + iloc2);
        fmax = (f1max > f2max) ? f1max : f2max;

      /* now convert relative thresh to absolute threshhold */

        thresh *= fmax;
        if (verbose) 
           printf ("flct: relative threshold in abs. units = %g\n", thresh);
           fflush(stdout);

        free (f1temp);
        free (f2temp);
    }
  }

  /* debug: output the two input images to file "f1f2.dat" */
/*
      write2images("f1f2.dat",f1,f2,nx,ny,transp);
*/

  /* Create velocity arrays vx,vy and the velocity mask array vm */

/* No longer create vx,vy,vm arrays in flct function */

/*
  vx = (double *) malloc (sizeof (double) * nx * ny);
  vy = (double *) malloc (sizeof (double) * nx * ny);
*/

  /* the vm array (velocity mask array, not to be confused with the
   * gaussian mask array that is used to modulate the images) 
   * will later be set to 1.0 where the velocity is computed,
   * and to 0.0 where the velocity is not computed -- because the image
   * pixels are below the noise threshold value "thresh" input from
   * the command line. */

/*
  vm = (double *) malloc (sizeof (double) * nx * ny);
*/

  /* Now create master gaussian image mask: */

  gaussdata = (double *) malloc (sizeof (double) * (2 * nxorig) * (2 * nyorig));
  gamx= (double *) malloc( sizeof(double) * nx * ny);
  gamy= (double *) malloc( sizeof(double) * nx * ny);
  gamx2= (double *) malloc( sizeof(double) * nx * ny);
  gamy2= (double *) malloc( sizeof(double) * nx * ny);
  gamxy= (double *) malloc( sizeof(double) * nx * ny);
  gampeak=(double *) malloc( sizeof(double) * nx * ny);

  if(!sigmaeq0) /* this case for sigma > 0 */
  {
    for (i = 0; i < 2 * nxorig; i++)
    {
        argx = sigminv * (double) (i - nxorig);
        for (j = 0; j < 2 * nyorig; j++)
        {
	  argy = sigminv * (double) (j - nyorig);
	  *(gaussdata + i * (2 * ny) + j) = exp (-argx * argx - argy * argy);
        }
    }
  }
  else /* this case for sigma = 0. ie set gaussian to 1.0 */
  {
    for (i = 0; i < 2 * nxorig; i++)
    {
        for (j = 0; j < 2 * nyorig; j++)
        {
	  *(gaussdata + i * (2 * nyorig) + j) = (double) 1.;
        }
    }
  }

      /* Debug -  output the gaussian image mask data: */
      /*
      writeimage("gaussdata.dat",gaussdata,2*nxorig,2*nyorig,transp);
      */

  gam2oversigma2=0.;
  nsig=0;

  /* Now do the master loop over i,j for computing velocity field: */

  for (i = 0; i < nx; i++)
    {
      for (j = 0; j < ny; j++)
	{
          if((nx ==1) && (ny == 1))
            {
              init=2; /* special case: must initialize AND destroy plans */
            }
	  else if ((i == 0) && (j == 0) && ((nx+ny) > 2))
	    {
	      /* 1st time through, set init to 1 so that
	       * fftw FFT plans are initialized */

	      init = 1;
	    }
	  else if ((i == (nx - 1)) && (j == (ny - 1)) && ((nx+ny) > 2))
	    {
	      /* last time through, set init to -1 so that
	       * fftw static variables are freed and plans
	       * destroyed */

	      init = -1;
	    }
	  else
	    {
	      /* the rest of the time just chunk along */

	      init = 0;
	    }

	  /* Now, figure out if image value is below
	   * threshold: */

	  /* the data is considered below theshhold if the
	   * absolute value of average of the pixel value from the 2 images 
	   * is below threshold */

	  fabsbar = 0.5 * (fabs (*(f1 + i * ny + j) + *(f2 + i * ny + j)));
	  belowthresh = (fabsbar < thresh);

	  /* Or alternatively:
	     belowthresh = ((fabs1 < thresh) || (fabs2 < thresh));
	   */

	  /* all the hard work of doing the cross-correlation
	   * needs to be done if the avg data is above the
	   * threshold OR if init != 0 */

          /* added skip logic here */

          skipon=skip+abs(qoffset)+abs(poffset);

          if(skipon)
          {
             if(transp)
             {
                xoffset=qoffset;
                yoffset=poffset;
             }
             else
             {
                xoffset=poffset;
                yoffset=qoffset;
             }
             noskipx = !((i-xoffset) % skip);
             noskipy = !((j-yoffset) % skip);
             noskipxy=noskipx*noskipy;
          }
          else
          {
             noskipxy=1;
          }

	  hardworkneeded = (((!belowthresh) && (noskipxy)) || (init != 0));

	  if (hardworkneeded)
	    {

	      /* the hard work for this particular pixel starts
	       * now */


	      /* Now find where the gaussian modulated image 
	       * is
	       * chopped off by the sliding box.  The sliding
	       * box is centered at i,j, unless the edges of
	       * the box would go outside the array -- 
	       * then the
	       * sliding box just sits at edges and/or corners
	       * of the array   */

              if(!sigmaeq0) /* for sigma > 0 */
              {
	        imin0 = (0 > (i - (nsize - 1))) ? 0 : i - (nsize - 1);
                imax0 = ((nx - 1) < (i + nsize)) ? nx - 1 : i + nsize;
                imin = (imax0 == nx - 1) ? nx - 1 - (2 * nsize - 1) : imin0;
                imax = (imin0 == 0) ? 0 + (2 * nsize - 1) : imax0;

                jmin0 = (0 > (j - (nsize - 1))) ? 0 : j - (nsize - 1);
                jmax0 = ((ny - 1) < (j + nsize)) ? ny - 1 : j + nsize;
                jmin = (jmax0 == ny - 1) ? ny - 1 - (2 * nsize - 1) : jmin0;
                jmax = (jmin0 == 0) ? 0 + (2 * nsize - 1) : jmax0;

                isize = imax - imin + 1;
                jsize = jmax - jmin + 1;

                 /* If the following tests for isize, jsize fail,
	         this is very bad:  exit */

	        if (isize != 2 * nsize)
                {
		  printf ("flct: exiting, bad isize = %d\n", isize);
		  exit (1);
                }
	        if (jsize != 2 * nsize)
                {
		  printf ("flct: exiting, bad jsize = %d\n", jsize);
		  exit (1);
                }
              }
              else /* if sigma = 0. just set isize=nxorig, jsize=nyorig */
              {
                 isize=nxorig;
                 jsize=nyorig;
                 imin=0;
                 jmin=0;
              }
              /* debug:
              printf("isize = %d, jsize = %d,\n",isize,jsize);
              */

              /* Compute sub-image means of f1 and f2: */

              f1bar=0.;
              f2bar=0.;
              for (ii = 0; ii < isize; ii++)
                { 
                   for (jj = 0; jj < jsize; jj++)
                      {
                         f1bar=f1bar+ *(f1 + (ii+imin)*nyorig + (jj+jmin));
                         f2bar=f2bar+ *(f2 + (ii+imin)*nyorig + (jj+jmin));
                      }
                }

              f1bar=f1bar/((double)isize*jsize);
              f2bar=f2bar/((double)isize*jsize);

	      g1 = (double *) malloc (sizeof (double) * isize * jsize);
	      g2 = (double *) malloc (sizeof (double) * isize * jsize);

/* comment out the g1bar calculation:  
   This is not being used, but code retained just in case it is resurrected. */

/*
              g1bar=0.;
              g2bar=0.;
              for (ii = 0; ii < isize; ii++)
                {
                  for (jj = 0; jj < jsize; jj++)
                    {
                       g1bar=g1bar+ *(g1tmp + (ii+imin)*nyorig
                        +(jj+jmin));
                       g2bar=g2bar+ *(g2tmp + (ii+imin)*nyorig
                        +(jj+jmin));
                    }
                }
              g1bar=g1bar/((double)isize*jsize);
              g2bar=g2bar/((double)isize*jsize);
*/

	      /* Now fill the reduced size arrays (sub-images) with the 
	       * appropriate values from the 
	       * full-sized arrays: */

	      for (ii = 0; ii < isize; ii++)
		{
		  for (jj = 0; jj < jsize; jj++)
		    {
		      *(g1 + ii * jsize + jj) = 
                          *(gaussdata + (nxorig-i+(ii+imin))*2*nyorig
                           +nyorig-j+(jj+jmin)) *
                          (*(f1 + (ii + imin) * nyorig + (jj + jmin))-f1bar) ;

		      *(g2 + ii * jsize + jj) = 
                          *(gaussdata + (nxorig-i+(ii+imin))*2*nyorig
                           +nyorig-j+(jj+jmin)) *
                          (*(f2 + (ii + imin) * nyorig + (jj + jmin))-f2bar) ;
		    }
		}

	      /* Call to cross_cor is used to find the 
	       * relative
	       * shift of image g2 relative to g1: */

	      icc = cross_cor (init, hires, expand, g1, g2, &absccor,
			       isize, jsize, &shiftx, &shifty, filter, 
                               kr, sigma);

/*                              debug:  output of absccor */

/*
                              writeimage("absccor.dat",absccor,isize,jsize,
                                 transp);
*/


              absccmax=1;
              maxind = maxloc (absccor, isize * jsize);
              if( *(absccor+maxind) == (double)0.)
              {
                 absccmax=0;
              }
              if(absccmax == 1)
              {
                 ixmax = maxind / jsize;
                 iymax = maxind % jsize;
              }
              else
              {
                 ixmax = isize/2;
                 iymax = jsize/2;
              }
              /* printf("flct: ixmax = %d, iymax = %d\n",ixmax,iymax); */
              gamx2val=0.;
              gamy2val=0.;
              gamxyval=0.;
              gampeakval=0.;
              gamxval=0.;
              gamyval=0.;
              if( (ixmax > 0) && (ixmax < (isize-1)) && (iymax > 0) && 
                  (iymax < (jsize-1)) && (absccmax == 1))
              {

                fx=0.5* ( *(absccor+(ixmax+1)*jsize+iymax) - 
                     *(absccor+(ixmax-1)*jsize+iymax) );
                fy=0.5* ( *(absccor+ixmax*jsize+iymax+1) - 
                   *(absccor+ixmax*jsize+iymax-1) );
                fxx = ( *(absccor+(ixmax+1)*jsize+iymax)+ 
                   *(absccor+(ixmax-1)*jsize+iymax) 
                  -2.*( *(absccor+ixmax*jsize+iymax))  );
                fyy = ( *(absccor+ixmax*jsize+iymax+1) + 
                   *(absccor+ixmax*jsize+iymax-1)
                  -2.*( *(absccor+ixmax*jsize+iymax)) );
                fxy = 0.25*( *(absccor+(ixmax+1)*jsize+iymax+1) + 
                   *(absccor+(ixmax-1)*jsize+iymax-1) -
                   *(absccor+(ixmax+1)*jsize+iymax-1) - 
                   *(absccor+(ixmax-1)*jsize+iymax+1) );
                fpeak=*(absccor+ixmax*jsize+iymax);
/*
                gamx2inv=-0.5*fxx/fpeak;
                gamy2inv=-0.5*fyy/fpeak;
                gamxyinv=fxy/fpeak;
*/
                gamx2val=fxx;
                gamy2val=fyy;
                gamxyval=fxy;
                gampeakval=fpeak;
                gamxval=fx;
                gamyval=fy;
                hessian=fxx*fyy/(fpeak*fpeak)-fxy*fxy/(fpeak*fpeak);
                hm1over2=1./sqrt(hessian);
                /* Don't let ratio of hm1over2 to sigma^2 approach 1 
                 or get singularity in corfac */
                if((hm1over2 > 0.95*sigma*sigma) || (hessian == 0))
                {
                  hm1over2=0.95*sigma*sigma;
                }
                if(!sigmaeq0 && (hessian > 0))
                {
                  /* develop statistics for mean value of gamma^2/sigma^2 */
                  gam2oversigma2+=(hm1over2/(sigma*sigma));
                  nsig++;
                }
                if( (sigmaeq0) || (!biascor)) 
                {
                  corfac=1.;
                }
                else 
                {
                  corfac=1./(1.-0.8*hm1over2/(sigma*sigma));
                }
                /* Now add corrections to shiftx, shifty 
                based on bias correction factor */
                shiftx*=corfac;
                shifty*=corfac;
              }

/* free temporary arrays created during loop */

	      free (g1);
	      free (g2);
	      free (absccor);

	      /* Now convert shifts to units of velocity using
	       * deltas and deltat */

	      /* Note: if (transp), then the meaning of 
	       * velocities
	       * has to be switched between x and y */

	      if (transp)
		{
		  vxx = shifty * deltinv * deltas;
		  vyy = shiftx * deltinv * deltas;
                  *(gamx2+i*ny+j)=gamy2val;
                  *(gamy2+i*ny+j)=gamx2val;
                  *(gamxy+i*ny+j)=gamxyval;
                  *(gampeak+i*ny+j)=gampeakval;
                  *(gamx+i*ny+j)=gamyval;
                  *(gamy+i*ny+j)=gamxval;
		}
	      else
		{
		  vxx = shiftx * deltinv * deltas;
		  vyy = shifty * deltinv * deltas;
                  *(gamx2+i*ny+j)=gamx2val;
                  *(gamy2+i*ny+j)=gamy2val;
                  *(gamxy+i*ny+j)=gamxyval;
                  *(gampeak+i*ny+j)=gampeakval;
                  *(gamx+i*ny+j)=gamxval;
                  *(gamy+i*ny+j)=gamyval;
		}

	      /* all the hard work for this pixel is now done */

	    }

	  /* default value for vmask is 1. */

	  vmask = 1.;

	  if ((belowthresh || !noskipxy) && !sigmaeq0)

	    /* If data below threshold, set vxx, vyy to xmiss 
	     * and vmask to 0, meaning vel not computed. */
            /* If sigma=0 just ignore the threshold and compute anyway */

	    {
	      vxx = xmiss;
	      vyy = xmiss;
	      vmask = 0.;
              *(gamx2 + i*ny +j)=0.;
              *(gamy2 + i*ny +j)=0.;
              *(gamxy + i*ny +j)=0.;
              *(gampeak +i*ny +j)=0.;
              *(gamx + i*ny +j)=0.;
              *(gamy + i*ny +j)=0.;
	    }


	  if ((j == 0) && (verbose))

	    {
	      printf ("flct: progress  i = %d out of %d\r", i, nx - 1);
	      fflush (stdout);
	    }

	  *(vx + i * ny + j) = vxx;
	  *(vy + i * ny + j) = vyy;
	  *(vm + i * ny + j) = vmask;
          if(verbose && sigmaeq0)
          {
              printf("\nflct: vx = %g vy = %g \n",vxx,vyy);
              fflush(stdout);
          }

	}
    }
   if(!sigmaeq0)
   {
/*   printf("flct debug: gam2oversigma2 = %g, nsig=%d\n",gam2oversigma2,nsig);*/
     gam2oversigma2/=nsig;
     if(verbose)
     {
       printf ("flct: mean value of gamma^2/sigma^2 = %g\n",gam2oversigma2);
       fflush(stdout);
     }
   }
/* Debug
    printf("\n"); 
*/

  /*  If interpolation to non-computed pixels is desired, this next code is
      where that is done */

  if(skipon && interpolate)
  {
  /* First step is to figure out the number of computed points in x,y 
  which we'll call ixcount and iycount */
  if(transp)
    {
       xoffset=qoffset;
       yoffset=poffset;
    }
  else
    {
       xoffset=poffset;
       yoffset=qoffset;
    }
     ixcount=0;
     for(i=0;i<nx;i++)
     {
        noskipx = !((i-xoffset) % skip);
        if(noskipx) ixcount++;
     }

     iycount=0;
     for(j=0;j<ny;j++)
     {
        noskipy = !((j-yoffset) % skip);
        if(noskipy) iycount++;
     }

  /* Now that we know ixcount,iycount create arrays big enough to hold the
  computed points.  Call these arrays vxnoi, vynoi, plus the mask, vmnoi */

  vxnoi=(void *)malloc(ixcount*iycount*sizeof(double));
  vynoi=(void *)malloc(ixcount*iycount*sizeof(double));
  vmnoi=(void *)malloc(ixcount*iycount*sizeof(double));
  }

  if(skipon && interpolate)
  /* Next step is to fill in the arrays vxnoi, vynoi with the computed
     values from the vy, vy arrays */
  {
  if(transp)
    {
       xoffset=qoffset;
       yoffset=poffset;
    }
  else
    {
       xoffset=poffset;
       yoffset=qoffset;
    }

     ic=-1;
     for(i=0;i<nx;i++)
     {
        noskipx = !((i-xoffset) % skip);
        if(noskipx) ic++;
        jc=-1;
        for(j=0;j<ny;j++)
        {
           noskipy = !((j-yoffset) % skip);
           if(noskipy) jc++;
           noskipxy=noskipx*noskipy; 
           if(noskipxy) *(vxnoi+ic*iycount+jc)=*(vx+i*ny+j);
           if(noskipxy) *(vynoi+ic*iycount+jc)=*(vy+i*ny+j);
           if(noskipxy) *(vmnoi+ic*iycount+jc)=*(vm+i*ny+j);
        }
     }
  }
  /* DEBUG:
  write3images ("noiarrays.dat", vxnoi,vynoi,vmnoi,ixcount,iycount,transp);
  */

  /* Next step is to compute xwant, ywant arrays, to get values of x,y at
     which interpolation is desired.  Note that sometimes these arrays
     will be outside the range of vxnoi, vynoi, especially if non-zero
     values of p,q are used. */

  if(skipon && interpolate)
  {
    xwant=(void *)malloc(nx*sizeof(double));
    ywant=(void *)malloc(ny*sizeof(double));

    for (i=0;i<nx;i++)
    {
    if(transp)
      {
         xoffset=qoffset;
      }
    else
      {
         xoffset=poffset;
      }
      *(xwant+i)=((double)(i-xoffset))/(double)(skip);
/* debug: 
      printf("for i= %d, xwant[i] = %g\n",i,*(xwant+i)); 
*/
    }

    for (j=0;j<ny;j++)
    {
    if(transp)
      {
         yoffset=poffset;
      }
    else
      {
         yoffset=qoffset;
      }
      *(ywant+j)=((double)(j-yoffset))/(double)(skip);
/* debug:
      printf("for j= %d, ywant[j] = %g\n",j,*(ywant+j)); 
*/

    }

/* Next step is to use interpcc2d function to interpolate vxnoi,vynoi arrays
   to vxint,vyint arrays.  Note vxint, vyint are allocated within interpcc2d
   and should be freed when you are done with them. */

      interpcc2d(vxnoi,xmiss,ixcount,iycount,xwant,nx,ywant,ny,&vxint);
      interpcc2d(vynoi,xmiss,ixcount,iycount,xwant,nx,ywant,ny,&vyint);

/* Debug: 
      write3images ("intarrays.dat", vxint,vyint,vyint,nx,ny,transp);
*/

/* Now put in the interpolated values of vxint, vyint back into the vx,vy arrays
   and set the vm array to 0.5 for interpolated values */

      for(i=0;i<nx;i++)
      {
        noskipx = !((i-xoffset) % skip);
        for(j=0;j<ny;j++)
         {
           if(transp)
           {
             xoffset=qoffset;
             yoffset=poffset;
           }
           else
           {
             xoffset=poffset;
             yoffset=qoffset;
           }
           noskipy = !((j-yoffset) % skip);
           noskipxy=noskipx*noskipy; 
           skipxy=!noskipxy;
           if(skipxy) *(vx+i*ny+j)=*(vxint+i*ny+j);
           if(skipxy) *(vy+i*ny+j)=*(vyint+i*ny+j);
           if(skipxy) *(vm+i*ny+j)=0.5;
         }
      }
     /* Now need to free all the arrays that have been created for
        the interpolation operation */
    free(vxnoi);
    free(vynoi);
    free(vmnoi);
    free(xwant);
    free(ywant);
    free(vxint);
    free(vyint);
    /* Should finally be done with the interpolation over skipped data */
  }

  /* Finally, reset any values of vx,vy that are equal to xmiss to zero, and
     make sure corresponding value of vm is also zero. */

  for(i=0;i<nx;i++)
   {
      for(j=0;j<ny;j++)
       {
          if(*(vx+i*ny+j) == xmiss) *(vm+i*ny+j)=0.;
          if(*(vx+i*ny+j) == xmiss) *(vx+i*ny+j)=0.;
          if(*(vy+i*ny+j) == xmiss) *(vy+i*ny+j)=0.;
       }
   }

  /* Outer loops over i,j finally done! */
  /* Output the vx, vy arrays to the output file 'outfile': */

  /* Output step now must be done in main program */
  /* write3images (outfile, vx, vy, vm, nx, ny, transp); */

  /* free the gaussian mask array, the original images, and the
   * velocity arrays */

  free (gaussdata);
#ifdef CCDATA
  write3images (gamfile, gamx2, gamy2, gamxy, nx, ny, transp);
  write3images (peakfile, gampeak, gamx, gamy, nx,ny,transp);
#endif
  free (gamx2);
  free (gamy2);
  free (gamxy);
  free (gampeak);
  free (gamx);
  free (gamy);

/* Don't want to free these things anymore, they get returned */
/*
  free (f1);
  free (f2);
  free (vx);
  free (vy);
  free (vm);
*/

  if (verbose)
  {
    printf ("\nflct: finished\n");
    fflush(stdout);
  }

  /* we're done! */
  return 0;
  /*  END FLCT FUNCTION */
}

i4 readimage (char *fname, i4 * nx, i4 * ny, double **arr,
	     i4 transp)
/* Function to read array dims, create space for 1 double array, read it in.
 * Note the double pointer to the double precision * arrays in the calling 
 * argument.  Note also these are referenced as pointers and returned to the
 * calling function. */
{
  FILE *f1;			/* pointer to input file */
  i4 newsize;			/* size of the new double prec. array to be 
				   read in =nx*ny */
  i4 i, ier, ibe, ise, vcid;
  f4 *farr;
  ibe = is_large_endian ();
  ise = 0;
  if (ibe == 0) ise = 1;	/* set small endian flag if not big  endian */
  f1 = fopen (fname, "rb");	/* open file for binary unformatted read */
  ier = 0;
  if (f1 == NULL)		/* error exit if file can't be opened */
    {
      printf ("readimage: cannot open file %s\n", fname);
      exit (1);
    }

  /* check that files begins with special vel_ccor flag: */
  fread (&vcid, sizeof (i4), 1, f1);
  if (ise) byteswapflct ((void *) &vcid, 1, sizeof (i4));
  if (vcid != 2136967593)
    {
      printf ("readimage: input file is not a vel_ccor i/o file\n");
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
      printf ("readimage: memory request for arr failed\n");
      exit (1);
    }
/*
	printf("%d bytes of memory now allocated for arr \n",newsize);
*/

  /* now read in the arr array */

  fread (farr, sizeof (f4), (*nx) * (*ny), f1);
  if (ise) byteswapflct ((void *) farr, (*nx) * (*ny), sizeof (f4));
  /*byteswap if needed */


  /* now transfer data from temp. arrays to arr: */

  for (i = 0; i < (*nx) * (*ny); i++)
    {
      *(*arr + i) = (double) *(farr + i);
    }

  /* now free the temp. arrays and close the files */

  free (farr);
  fclose (f1);
  ier = 1;
  return ier;
}

i4 read2images (char *fname, i4 * nx, i4 * ny, double **arr, double **barr,
	     i4 transp)
/* Function to read array dims, create space for 2 double arrays, read them in.
 * Note the double pointer to the double precision * arrays in the calling 
 * argument.  Note also these are referenced as pointers and returned to the
 * calling function. */
{
  FILE *f1;			/* pointer to input file */
  i4 newsize;			/* size of the new double prec. array to be 
				   read in =nx*ny */
  i4 i, ier, ibe, ise, vcid;
  f4 *farr, *fbarr;
  ibe = is_large_endian ();
  ise = 0;
  if (ibe == 0) ise = 1;	/* set small endian flag if not big  endian */
  f1 = fopen (fname, "rb");	/* open file for binary unformatted read */
  ier = 0;
  if (f1 == NULL)		/* error exit if file can't be opened */
    {
      printf ("read2images: cannot open file %s\n", fname);
      exit (1);
    }

  /* check that files begins with special vel_ccor flag: */
  fread (&vcid, sizeof (i4), 1, f1);
  if (ise) byteswapflct ((void *) &vcid, 1, sizeof (i4));
  if (vcid != 2136967593)
    {
      printf ("read2images: input file is not a vel_ccor i/o file\n");
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
      printf ("read2images: memory request for arr failed\n");
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
      printf ("read2images: memory request for barr failed\n");
      exit (1);
    }

  /* now read in the fbarr array */

  fread (fbarr, sizeof (f4), (*nx) * (*ny), f1);
  /*byteswap if needed */
  if (ise) byteswapflct ((void *) fbarr, (*nx) * (*ny), sizeof (f4));

  /* now transfer data from temp. arrays to arr and barr: */

  for (i = 0; i < (*nx) * (*ny); i++)
    {
      *(*arr + i) = (double) *(farr + i);
      *(*barr + i) = (double) *(fbarr + i);
    }

  /* now free the temp. arrays and close the files */

  free (farr);
  free (fbarr);
  fclose (f1);
  ier = 1;
  return ier;
}

i4 where (char *cond, i4 xsize, i4 ** index, i4 * length_index)
/* This function serves a similar purpose to the "where" function in IDL.
 * Given the array *cond (char) of length xsize 
 * containing a pre-computed condition, the function finds **index, a double
 * pointer to an array of indices which reference those values
 * where *cond is
 * non-zero.  The integer *length_index returns the length of *index. 
 */
/* Note - this function no longer used in vel_ccor */
{
  i4 ier;	/* function return value - not thought through yet */
  i4 i, ii;	/* counter variables */
  i4 *indtmp;	/* temporary local array of indices of *x */
  ier = 0;	/*return value of function */
  *length_index = 0;	/* initialize length of *index array to 0 */

/*	printf("\nxsize = %d",xsize); */

  indtmp = (i4 *) malloc (sizeof (i4) * xsize);	/* create temp. ind. array */

  /* Ready to start */

  ii = 0;		/* set initial counter of temp index array to 0 */
  for (i = 0; i < xsize; i++)	/* start incrementing the *cond array: */
    {
      if (*(cond + i))
	{
	  /* if the condition is true, record i into temp index */
	  *(indtmp + ii) = (i4) i;
	  ii++;		/* and then increment ii */
	}
      /* otherwise just keep incrementing i and doing nothing */
    }

/*	printf("\nii= %d\n", ii) ;
	fflush (stdout) ; */

  /* Now create right amount of space for *index: */

  *index = (i4 *) malloc (sizeof (i4) * ii);

  /* Now copy index values from temp array into *index array */

  memcpy ((void *) *index, (void *) indtmp, ii * sizeof (i4));

  /* Now set the length of the *index array */

  *length_index = (i4) ii;

  /* Now free memory from temp. index array */

  free (indtmp);
  return ier;			/* always 0 at the moment */
}

i4 cross_cor (i4 init, i4 hires, i4 expand, double *arr, double *barr,
	   double **absccor, i4 nx, i4 ny, double *shiftx, double *shifty, 
           i4 filterflag, double kr, double sigma)
{
  i4 i, j, ixx, iyy, maxind, ixmax, iymax, ishft, maxfine, absccmax;
  i4 nxinterp, nyinterp, nfgppergp;
  double normfac, rangex, rangey, shiftx0, shifty0, shiftxx, shiftyy;
  double *xwant, *ywant, *peakarea;
  double shiftsubx, shiftsuby, fx, fy, fxx, fyy, fxy, fpeak;
  double gamx2inv,gamy2inv,sigminv,corx,cory;
  double xmiss=0.;

  /* following variables must be saved between calls; declared static */

  static double *ina, *inb, *ccor;
  static double *filter, *kx, *ky;
  static fftw_complex *outa, *outb, *ccorconj;
  static fftw_plan pa, pb, pback;

  /* absccor is a double pointer containing abs. value of cc function */

  /* debug:
  printf("cross_cor: nx = %d, ny = %d\n",nx,ny);
  */

  *absccor = malloc (sizeof (double) * nx * ny);
  /* get sigminv = 1./sigma , if sigma==0 set to 0 */

  /*  Following stuff is obsolete; should probably be deleted. */
  sigminv=0.;
  if(sigma !=0) sigminv=1./sigma;
  corx=1.0;
  cory=1.0;
  gamx2inv=1.0;
  gamy2inv=1.0;
  /* end of obsolete stuff */

  /* Set up interpolation grid depending on whether hires set or not */

  if (hires == 1)
    {
      nxinterp = 101;
      nyinterp = 101;
      nfgppergp = 50;
    }
  else
    {
      nxinterp = 21;
      nyinterp = 21;
      nfgppergp = 10;
    }
/*	printf("initialization stuff done in cross_cor\n"); */
  if ((init == 1) || (init == 2))
    {
      /* First time through: */
      /* Initialization of FFT variables and FFTW plans. */
      /* NOTE -- empirically had to add 1 to "y" dimensions of outa,
       * outb, and ccorconj to
       * avoid a memory leak and a seg fault at fftw_free */

      /* should check to see if still a problem */

      outa = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) *
					   nx * ((ny / 2) + 2));
      outb = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) * 
              nx * ((ny / 2) + 2));	/* valgrind sometimes complains */
      ccorconj = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) *
					       nx * ((ny / 2) + 2));

      ina = (double *) fftw_malloc (sizeof (double) * nx * ny);
      inb = (double *) fftw_malloc (sizeof (double) * nx * ny);
      ccor = (double *) fftw_malloc (sizeof (double) * nx * ny);
      filter = (double *) fftw_malloc (sizeof (double) * nx * ny);
      kx=(double *) fftw_malloc (sizeof(double)*nx);
      ky=(double *) fftw_malloc (sizeof(double)*ny);
      if(filterflag)
      {
         make_freq(kx,nx);
         make_freq(ky,ny);
         gaussfilt(filter,kx,ky,nx,ny,kr);
      }

      for (i = 0; i < nx * ny; i++)
	{
	  *(ina + i) = (double) 0.;
	  *(inb + i) = (double) 0.;
	}
      for (i = 0; i < nx * ((ny / 2 + 1)); i++)
      {
#ifdef COMPLEXH
        /* If complex.h included, do this: */
        *(ccorconj+i)=0.+I*0.;
#else
        /* If complex.h not included, do this: */
	ccorconj[i][0] = 0.;
	ccorconj[i][1] = 0.;
#endif
      }
      pa = fftw_plan_dft_r2c_2d (nx, ny, ina, outa, FFTW_MEASURE);
      pb = fftw_plan_dft_r2c_2d (nx, ny, inb, outb, FFTW_MEASURE);
      pback = fftw_plan_dft_c2r_2d (nx, ny, ccorconj, ccor, FFTW_MEASURE);
    }

    for (i = 0; i < nx * ny; i++)

    {
/*		printf("1st loop: i = %d, *(arr+i)= %g, *(barr+i) = %g\n",
				i,*(arr+i),*(barr+i)); */

      /* copy from input doubles to fftw variables */

      *(ina + i) = (double) (*(arr + i));
      *(inb + i) = (double) (*(barr + i));
    }

  /* actually do the forward FFTs: */

  fftw_execute (pa);
  fftw_execute (pb);

  /* calculate normalization factor */

  normfac = (1. / ((double) nx * ny));
  normfac *= normfac; /* square of above line */
 
/* Now apply the gaussian filter to the FFT'd data in frequency space */

    if(filterflag)
    {
      for (i=0;i<nx;i++)
      {
         for (j=0;j<(ny/2)+1;j++)
         {
#ifdef COMPLEXH
/* If complex.h is invoked, just multiply outa and outb by filter function */
            outa[i*((ny/2)+1)+j]=outa[i*((ny/2)+1)+j]*filter[i*ny+j];
            outb[i*((ny/2)+1)+j]=outb[i*((ny/2)+1)+j]*filter[i*ny+j];
#else
/* If complex.h not invoked, multiply real and im parts by filter function: */
            outa[i*((ny/2)+1)+j][0]=outa[i*((ny/2)+1)+j][0]*filter[i*ny+j];
            outa[i*((ny/2)+1)+j][1]=outa[i*((ny/2)+1)+j][1]*filter[i*ny+j];
            outb[i*((ny/2)+1)+j][0]=outb[i*((ny/2)+1)+j][0]*filter[i*ny+j];
            outb[i*((ny/2)+1)+j][1]=outb[i*((ny/2)+1)+j][1]*filter[i*ny+j];
#endif            
         }
      }
    }

  /* Now calculate product of conj(outa) * outb */

  for (i = 0; i < nx * ((ny/2) + 1); i++)
    {

#ifdef COMPLEXH
        /* if complex.h included, do this */
        *(ccorconj+i)=(conj(*(outa+i))*(*(outb+i)))*normfac;
#else
        /* if complex.h not invoked, do this: */
        ccorconj[i][0] = (outa[i][0] * outb[i][0] + outa[i][1] * outb[i][1])
	* normfac;
        ccorconj[i][1] = (outa[i][0] * outb[i][1] - outa[i][1] * outb[i][0])
	* normfac;
#endif
    }

  /* now do the inverse transform to get cc function */

  fftw_execute (pback);

  /* now calculate the absolute value of cc function */

  for (i = 0; i < nx * ny; i++)
    {
      *(*absccor + i) = (double) fabs(*(ccor+i));
    }

  if ((init == -1) || (init == 2))
    {
      /* Last time through: free all the plans and static variables */

      fftw_free (outa);
      fftw_free (outb);
      fftw_free (ccorconj);
      fftw_free (ccor);
      fftw_free (filter);
      fftw_free (kx);
      fftw_free (ky);
      fftw_free (ina);
      fftw_free (inb);
      fftw_destroy_plan (pa);
      fftw_destroy_plan (pback);
      fftw_destroy_plan (pb);
    }

/* Now shift the absccor array by nx/2, ny/2 to avoid aliasing problems */

  ishft = shift2d (*absccor, nx, ny, nx / 2, ny / 2);

  /* Now find maximum of the shifted cross-correlation function to 1 pixel
     accuracy:  */

  absccmax=1;
  maxind = maxloc (*absccor, nx * ny);
  if( *(*absccor+maxind) == (double)0.) 
  {
     absccmax=0;
  }
  if(absccmax == 1)
  {
     ixmax = maxind / ny;
     iymax = maxind % ny;
  }
  else
  {
     ixmax = nx/2;
     iymax = ny/2;
  }
  shiftx0 = ixmax;
  shifty0 = iymax;
  shiftsubx=0.;
  shiftsuby=0.;

  if(( expand == 1) && (hires == -1) && (ixmax > 0) && (ixmax < (nx-1))
     && (iymax > 0) && (iymax < (ny-1)) && (absccmax == 1))
  {
     fx=0.5* ( *(*absccor+(ixmax+1)*ny+iymax) - 
         *(*absccor+(ixmax-1)*ny+iymax) );
     fy=0.5* ( *(*absccor+ixmax*ny+iymax+1) - *(*absccor+ixmax*ny+iymax-1) );
     fxx = ( *(*absccor+(ixmax+1)*ny+iymax)+ *(*absccor+(ixmax-1)*ny+iymax)
        -2.*( *(*absccor+ixmax*ny+iymax))  );
     fyy = ( *(*absccor+ixmax*ny+iymax+1) + *(*absccor+ixmax*ny+iymax-1)
        -2.*( *(*absccor+ixmax*ny+iymax)) );
     fxy = 0.25*( *(*absccor+(ixmax+1)*ny+iymax+1) + 
            *(*absccor+(ixmax-1)*ny+iymax-1) -
            *(*absccor+(ixmax+1)*ny+iymax-1) - 
            *(*absccor+(ixmax-1)*ny+iymax+1) );
     fpeak=*(*absccor+ixmax*ny+iymax);
/* In following expressions for subshifts, shift is in units of pixel length */
     shiftsubx=(fyy*fx-fy*fxy)/(fxy*fxy-fxx*fyy);
     shiftsuby=(fxx*fy-fx*fxy)/(fxy*fxy-fxx*fyy);
  }

  shiftxx=shiftx0 + shiftsubx;
  shiftyy=shifty0 + shiftsuby;
/*
       printf("shiftx0-nx/2 = %g\n",(shiftx0-(double)(nx/2)));
       printf("shifty0-ny/2 = %g\n",(shifty0-(double)(ny/2)));
 
*/

/* Now create x, y arrays to define desired interpolation grid: */
  if(hires != -1)
  {

     rangex = (double) (nxinterp - 1) / nfgppergp;
     rangey = (double) (nyinterp - 1) / nfgppergp;

     xwant = (double *) malloc (sizeof (double) * nxinterp);
     ywant = (double *) malloc (sizeof (double) * nyinterp);

     for (i = 0; i < nxinterp; i++)
       {
         *(xwant + i) = ((((double) i) * rangex) / ((double) (nxinterp - 1)))
	   - 0.5 * rangex + shiftx0;
/*                 printf("xwant[%d] = %g\n",i,*(xwant+i)); */
       }
     for (j = 0; j < nyinterp; j++)
       {
         *(ywant + j) = ((((double) j) * rangey) / ((double) (nyinterp - 1)))
   	- 0.5 * rangey + shifty0;
/*                 printf("ywant[%d] = %g\n",j,*(ywant+j)); */
       }
   
  /* Now, do the interpolation of the region of the peak of the cc fn */

     interpcc2d (*absccor, xmiss, nx, ny, xwant, nxinterp, ywant, 
          nyinterp, &peakarea);

  /* Following writeimage stmt is available if you need to examine the
   * peakarea array for debugging - note transpose of x,y for IDL  read */

/*
      transp=1;
      writeimage("peakarea.dat",peakarea,nxinterp,nyinterp,transp);
*/

  /* Now find the peak of the interpolated function */

     maxfine = maxloc (peakarea, nxinterp * nyinterp);
     ixx = maxfine / nyinterp;
     iyy = maxfine % nyinterp;
/* Here is where to compute sub-pixel shifts in peakarea if they're wanted */
     shiftsubx=0.;
     shiftsuby=0.;
     if((expand == 1) && (ixx > 0) && (ixx < (nxinterp-1)) && (iyy > 0)
        && (iyy < (nyinterp-1)))
     {
        fx=0.5* ( *(peakarea+(ixx+1)*nyinterp+iyy) - 
              *(peakarea+(ixx-1)*nyinterp+iyy) );
        fy=0.5* ( *(peakarea+ixx*nyinterp+iyy+1) - 
              *(peakarea+ixx*nyinterp+iyy-1) );
        fxx = ( *(peakarea+(ixx+1)*nyinterp+iyy)+ 
              *(peakarea+(ixx-1)*nyinterp+iyy)
              -2.*( *(peakarea+ixx*nyinterp+iyy))  );
        fyy = ( *(peakarea+ixx*nyinterp+iyy+1) + *(peakarea+ixx*nyinterp+iyy-1)
           -2.*( *(peakarea+ixx*nyinterp+iyy)) );
        fxy = 0.25*( *(peakarea+(ixx+1)*nyinterp+iyy+1) +
               *(peakarea+(ixx-1)*nyinterp+iyy-1) -
               *(peakarea+(ixx+1)*nyinterp+iyy-1) -
               *(peakarea+(ixx-1)*nyinterp+iyy+1) );
/* In following expressions for subshifts, must mpy by unit of pixel length */
        shiftsubx=((fyy*fx-fy*fxy)/(fxy*fxy-fxx*fyy))*
          rangex/((double) (nxinterp -1));
        shiftsuby=((fxx*fy-fx*fxy)/(fxy*fxy-fxx*fyy))*
          rangey/((double) (nyinterp -1));
     }
     shiftxx = *(xwant + ixx) + shiftsubx;
     shiftyy = *(ywant + iyy) + shiftsuby;
  /* Free the variables created during interpolation */
     free (xwant);
     free (ywant);
     free (peakarea);
  }

/* Now, assign values to shiftx, shifty to return to calling program */

  *shiftx = shiftxx - (double) (nx / 2);
  *shifty = shiftyy - (double) (ny / 2);

/* Following expressions used if only 1 pixel accuracy needed from absccor 
 *
	*shiftx=((double)ixmax)-(double)(nx/2);
	*shifty=((double)iymax)-(double)(ny/2);
*/

  return 0;
}

i4 make_freq(double *k, i4 ndim)
{
/* k is assumed already allocated in main program, with dimension ndim */
i4 n21,i,inext;
n21=(ndim/2)-1;
for (i=0;i<n21+1;i++)
{
	k[i]=(double)i;
}

inext=n21+1;
if((ndim/2)*2 != ndim)
{
	k[inext]=(double)(ndim/2);
	inext++;
        k[inext]=-(double)(ndim/2);
        inext++;
}

else
{
	k[inext]=(double)(ndim/2);
        inext++;
}

for (i=inext;i<ndim;i++)
{
	k[i]=-(double)(n21-(i-inext));
}
/* debug */

/*
for (i=0;i<ndim;i++)
{
	printf("i = %d, k = %g\n",i,k[i]);
}
*/

/* end debug */

return 0;
}

i4 gaussfilt(double *filter, double *kx, double *ky, i4 nx, i4 ny, double kr)
{
/* Assumes kx of size nx, ky of size ny, and filter of size (nx,ny) */
double kxmax,kymax,kxroll,kyroll,smxinv,smyinv,argx,argy;
i4 i,j;
kxmax=(double)kx[nx/2];
kymax=(double)ky[ny/2];
kxroll=kr*kxmax;
kyroll=kr*kymax;
smxinv=(double)1./kxroll;
smyinv=(double)1./kyroll;
for (i=0;i<nx;i++)
{
	argx=kx[i]*smxinv;
	for(j=0;j<ny;j++)
	{
                argy=ky[j]*smyinv;
		filter[i*ny+j]=exp( -(argx*argx + argy*argy) );
	}
}
return 0;
}

i4 filter_image(double *arr, double *barr, double *outarr, double *outbarr,
        i4 nx, i4 ny, double kr)

/* Takes images arr, barr and filters them by gaussians in k-space of width
kr*kmax, where 0 < kr < 1, and kmax is the maximum wavenumber in x and y,
considered separately.  The input arrays are arr, barr, and the output
arrays are outarr, and outbarr.  They are assumed already allocated in
caller.  

This function is not used in this particular version of flct (filtering
is done within cross_cor), but is
included as it may be useful in the future.

*/

{

  i4 i,j;
  double *ina, *inb;
  double *filter, *kx, *ky;
  double normfac;
  fftw_complex *outa, *outb;
  fftw_plan pa, pb, pbacka, pbackb;
  outa = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) *
     nx * ((ny / 2) + 2));
  outb = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) * 
     nx * ((ny / 2) + 2));	/* valgrind sometimes complains */
  ina = (double *) fftw_malloc (sizeof (double) * nx * ny);
  inb = (double *) fftw_malloc (sizeof (double) * nx * ny);
  filter = (double *) fftw_malloc (sizeof (double) * nx * ny);
  kx=(double *) fftw_malloc (sizeof(double)*nx);
  ky=(double *) fftw_malloc (sizeof(double)*ny);
  make_freq(kx,nx);
  make_freq(ky,ny);
  gaussfilt(filter,kx,ky,nx,ny,kr);
      for (i = 0; i < nx * ny; i++)
	{
	  *(ina + i) = (double) 0.;
	  *(inb + i) = (double) 0.;
	}
      for (i = 0; i < nx * ((ny / 2 + 1)); i++)
      {
#ifdef COMPLEXH
        /* If complex.h included, do this: */
        *(outa+i)=0.+I*0.;
        *(outb+i)=0.+I*0.;
#else
        /* if complex.h not included, do this: */
	outa[i][0] = 0.;
	outa[i][1] = 0.;
	outb[i][0] = 0.;
	outb[i][1] = 0.;
#endif
      }
      /* set up plans for FFTs: */
      pa = fftw_plan_dft_r2c_2d (nx, ny, ina, outa, FFTW_MEASURE);
      pb = fftw_plan_dft_r2c_2d (nx, ny, inb, outb, FFTW_MEASURE);
      pbacka = fftw_plan_dft_c2r_2d (nx, ny, outa, ina, FFTW_MEASURE);
      pbackb = fftw_plan_dft_c2r_2d (nx, ny, outb, inb, FFTW_MEASURE);

    for (i = 0; i < nx * ny; i++)

    {
/*		printf("1st loop: i = %d, *(arr+i)= %g, *(barr+i) = %g\n",
				i,*(arr+i),*(barr+i)); */

      /* copy from input doubles to fftw variables */

      *(ina + i) = (double) (*(arr + i));
      *(inb + i) = (double) (*(barr + i));
    }

  /* actually do the forward FFTs: */

  fftw_execute (pa);
  fftw_execute (pb);
 /* calculate normalization factor */

  normfac = (1. / ((double) nx * ny));

/* Now apply the gaussian filter to the FFT'd data in frequency space */

    for (i=0;i<nx;i++)
    {
      for (j=0;j<(ny/2)+1;j++)
      {
#ifdef COMPLEXH
        /* if complex.h included: */
        outa[i*((ny/2)+1)+j]=outa[i*((ny/2)+1)+j]*filter[i*ny+j]*normfac;
        outb[i*((ny/2)+1)+j]=outb[i*((ny/2)+1)+j]*filter[i*ny+j]*normfac;
#else
        /* If complex.h not included: */
        outa[i*((ny/2)+1)+j][0]=outa[i*((ny/2)+1)+j][0]*filter[i*ny+j]*normfac;
        outa[i*((ny/2)+1)+j][1]=outa[i*((ny/2)+1)+j][1]*filter[i*ny+j]*normfac;
        outb[i*((ny/2)+1)+j][0]=outb[i*((ny/2)+1)+j][0]*filter[i*ny+j]*normfac;
        outb[i*((ny/2)+1)+j][1]=outb[i*((ny/2)+1)+j][1]*filter[i*ny+j]*normfac;
#endif

      }
    }

/* now do the inverse transform to get filtered images: */

    fftw_execute (pbacka);
    fftw_execute (pbackb);

  for (i = 0; i < nx * ny; i++)
    {
      *(outarr+i)=(double) (*(ina+i));
      *(outbarr+i)=(double) (*(inb+i));
    }

/* Free the plans and locally created arrays */

      fftw_free (outa);
      fftw_free (outb);
      fftw_free (filter);
      fftw_free (kx);
      fftw_free (ky);
      fftw_free (ina);
      fftw_free (inb);
      fftw_destroy_plan (pa);
      fftw_destroy_plan (pbacka);
      fftw_destroy_plan (pb);
      fftw_destroy_plan (pbackb);

return 0;

}

i4 writeimage (char *fname, double *arr, i4 nx, i4 ny, i4 transp)
{

/* Function to write array dimensions, and then write out array */

  FILE *f1;
  i4 newsize;
  i4 i, ier, ibe, ise, vcid, vcidtmp;
  f4 *farr;
  i4 nxtmp, nytmp;
  vcid = 2136967593;
  vcidtmp = vcid;
  nxtmp = nx;
  nytmp = ny;
  ibe = is_large_endian ();
  ise = 0;
  if (ibe == 0)
    ise = 1;	/* set flag for byteswapping if small endian */

  /* get the file fname open for binary write */

  f1 = fopen (fname, "wb");
  ier = 0;
  if (f1 == NULL)
    {
      printf ("writeimage: cannot open file %s\n", fname);
      exit (1);
    }

  if (ise)
    {
      byteswapflct ((void *) &vcidtmp, 1, sizeof (i4));
      byteswapflct ((void *) &nxtmp, 1, sizeof (i4));
      byteswapflct ((void *) &nytmp, 1, sizeof (i4));
    }

  fwrite (&vcidtmp, sizeof (i4), 1, f1);  /* write vel_ccor id integer 1st */

  /* shift role of nx, ny if transp is set (for use of data w/ IDL */

  if (transp)
    {
      fwrite (&nytmp, sizeof (i4), 1, f1);
      fwrite (&nxtmp, sizeof (i4), 1, f1);
    }
  else
    {
      fwrite (&nxtmp, sizeof (i4), 1, f1);
      fwrite (&nytmp, sizeof (i4), 1, f1);
    }


/*
      printf("\n\nnx,ny wrote out to file arr = %d,%d\n",nx,ny);
*/
  newsize = nx * ny * sizeof (double);

  /* create temporary f4 array to write out */

  farr = (f4 *) malloc (sizeof (f4) * nx * ny);

  /* now fill the array */

  for (i = 0; i < nx * ny; i++)
    {
      *(farr + i) = (f4) * (arr + i);
    }

  if (ise)
    byteswapflct ((void *) farr, nx * ny, sizeof (f4));
  /* byteswap if small endian */

  /* write out the array */
  fwrite (farr, sizeof (f4), nx * ny, f1);

  /* free temp. array and close file */
  free (farr);
  fclose (f1);
  ier = 1;
  return ier;
}

i4 write2images (char *fname, double *arr, double *barr, 
         i4 nx, i4 ny, i4 transp)
{

/* Function to write array dimensions, and write out 2 arrays arr and barr
 * while converting them to single precision f4 arrays  */

  FILE *f1;
  i4 newsize;
  i4 ier, i, ise, ibe;
  i4 nxtmp, nytmp, vcid, vcidtmp;
  f4 *farr, *fbarr;
  nxtmp = nx;
  nytmp = ny;
  vcid = 2136967593;
  vcidtmp = vcid;
  ibe = is_large_endian ();
  ise = 0;
  if (ibe == 0) ise = 1;
  if (ise)		/* if small endian, byteswap nxtmp and nytmp */
    {
      byteswapflct ((void *) &vcidtmp, 1, sizeof (i4));
      byteswapflct ((void *) &nxtmp, 1, sizeof (i4));
      byteswapflct ((void *) &nytmp, 1, sizeof (i4));
    }

  /* open the file fname for a binary write */

  f1 = fopen (fname, "wb");
  ier = 0;
  if (f1 == NULL)
    {
      printf ("write2images: cannot open file %s\n", fname);
      exit (1);
    }

  fwrite (&vcidtmp, sizeof (i4), 1, f1);	/* write vel_ccor id flag */

  /* switch role of nx, ny if transp is set (for interaction with IDL) 
   * and write these into the file */

  if (transp)
    {
      fwrite (&nytmp, sizeof (i4), 1, f1);
      fwrite (&nxtmp, sizeof (i4), 1, f1);
    }
  else
    {
      fwrite (&nxtmp, sizeof (i4), 1, f1);
      fwrite (&nytmp, sizeof (i4), 1, f1);
    }

/*
      printf("\n\nnx,ny wrote out to file arr = %d,%d\n",nx,ny);
*/
  newsize = nx * ny * sizeof (double);

  /* create space for the temporary f4 arrays farr and fbarr */

  farr = (f4 *) malloc (sizeof (f4) * nx * ny);
  fbarr = (f4 *) malloc (sizeof (f4) * nx * ny);

  /* fill the temporary arrays */

  for (i = 0; i < nx * ny; i++)
    {
      *(farr + i) = (f4) * (arr + i);
      *(fbarr + i) = (f4) * (barr + i);
    }

  /* now write out the 2 arrays */

  if (ise)	/* byteswap if small endian */
    {
      byteswapflct ((void *) farr, nx * ny, sizeof (f4));
      byteswapflct ((void *) fbarr, nx * ny, sizeof (f4));
    }

  fwrite (farr, sizeof (f4), nx * ny, f1);
  fwrite (fbarr, sizeof (f4), nx * ny, f1);

  /* free temp arrays and close file */

  free (farr);
  free (fbarr);
  fclose (f1);
  ier = 1;
  return ier;
}

i4 write3images (char *fname, double *arr, double *barr, double *carr,
	      i4 nx, i4 ny, i4 transp)
{

/* Function to write array dimensions, and write out 3 arrays arr,barr, and carr
 * while converting them to single precision f4 arrays  */

  FILE *f1;
  i4 newsize;
  i4 ier, i, ibe, ise;
  i4 nxtmp, nytmp, vcid, vcidtmp;
  f4 *farr, *fbarr, *fcarr;
  nxtmp = nx;
  nytmp = ny;
  vcid = 2136967593;
  vcidtmp = vcid;
  ibe = is_large_endian ();
  ise = 0;
  if (ibe == 0) ise = 1;	/* test for small endian for doing byteswaps */
  if (ise)			/* byteswap nxtmp, nytmp if small endian */
    {
      byteswapflct ((void *) &vcidtmp, 1, sizeof (i4));
      byteswapflct ((void *) &nxtmp, 1, sizeof (i4));
      byteswapflct ((void *) &nytmp, 1, sizeof (i4));
    }

  /* open the file fname for a binary write */

  f1 = fopen (fname, "wb");
  ier = 0;
  if (f1 == NULL)
    {
      printf ("write3images: cannot open file %s\n", fname);
      exit (1);
    }

  fwrite (&vcidtmp, sizeof (i4), 1, f1);

  /* switch role of nx, ny if transp is set (for interaction with IDL) 
   * and write these into the file */

  if (transp)
    {
      fwrite (&nytmp, sizeof (i4), 1, f1);
      fwrite (&nxtmp, sizeof (i4), 1, f1);
    }
  else
    {
      fwrite (&nxtmp, sizeof (i4), 1, f1);
      fwrite (&nytmp, sizeof (i4), 1, f1);
    }

/*
      printf("\n\nnx,ny wrote out to file arr = %d,%d\n",nx,ny);
*/
  newsize = nx * ny * sizeof (double);

  /* create space for the temporary f4 arrays farr, fbarr, and fcarr */

  farr = (f4 *) malloc (sizeof (f4) * nx * ny);
  fbarr = (f4 *) malloc (sizeof (f4) * nx * ny);
  fcarr = (f4 *) malloc (sizeof (f4) * nx * ny);

  /* fill the temporary arrays */

  for (i = 0; i < nx * ny; i++)
    {
      *(farr + i) = (f4) * (arr + i);
      *(fbarr + i) = (f4) * (barr + i);
      *(fcarr + i) = (f4) * (carr + i);
    }

  if (ise)			/* if small endian, byteswap the arrays */
    {
      byteswapflct ((void *) farr, nx * ny, sizeof (f4));
      byteswapflct ((void *) fbarr, nx * ny, sizeof (f4));
      byteswapflct ((void *) fcarr, nx * ny, sizeof (f4));
    }

  /* now write out the 3 arrays */

  fwrite (farr, sizeof (f4), nx * ny, f1);
  fwrite (fbarr, sizeof (f4), nx * ny, f1);
  fwrite (fcarr, sizeof (f4), nx * ny, f1);

  /* free temp arrays and close file */

  free (farr);
  free (fbarr);
  free (fcarr);
  fclose (f1);
  ier = 1;
  return ier;
}

i4 shift2d (double *arr, i4 nx, i4 ny, i4 ishift, i4 jshift)
{

/* Circular shift of the x,y indices of array *arr by ishift,jshift */
/* This function is similar to the shift function in IDL.  nx, ny
 * are the assumed dimensions of the array */

  double *temp;
  i4 i, j, ii, jj;
  temp = (double *) malloc (sizeof (double) * nx * ny);
  for (i = 0; i < nx; i++)
    {
      ii = (i + ishift) % nx;	/* ii = (i + ishift) modulo nx */

      for (j = 0; j < ny; j++)
	{
	  jj = (j + jshift) % ny;	/* jj = (j+jshift) modulo ny */

	  /* Now members of temp array get shifted: */

	  *(temp + ii * ny + jj) = *(arr + i * ny + j);
	}
    }

  /* Now copy temp array back into arr, then destroy temp and return */

  memcpy ((void *) arr, (void *) temp, nx * ny * sizeof (double));
  free (temp);
  return 0;
}

i4 maxloc (double *arr, i4 xsize)
{

/* finds the location of the maximum of the double array *arr and returns it. */

  i4 i, location;
  double amax;
  /* initialize amax and location to 0th element */
  amax = *(arr + 0);
  location = 0;
  for (i = 1; i < xsize; i++)
    {
      if (*(arr + i) > amax)
	{
	  amax = *(arr + i);
	  location = i;
	}
    }
  return location;
}

i4 imaxloc (i4 * arr, i4 xsize)
{

/* finds the location of the maximum of the i4 array *arr and returns it. */

  i4 i, location;
  i4 amax;
  /* initialize amax and location to 0th element */
  amax = *(arr + 0);
  location = 0;
  for (i = 1; i < xsize; i++)
    {
      if (*(arr + i) > amax)
	{
	  amax = *(arr + i);
	  location = i;
	}
    }
  return location;
}

i4 minloc (double *arr, i4 xsize)
{

/* finds the location of the minimum of the double array *arr and returns it. */

  i4 i, location;
  double amin;
  /* initialize amin and location to 0th element */
  amin = *(arr + 0);
  location = 0;
  for (i = 1; i < xsize; i++)
    {
      if (*(arr + i) < amin)
	{
	  amin = *(arr + i);
	  location = i;
	}
    }
  return location;
}

i4 iminloc (i4 * arr, i4 xsize)
{

/* finds the location of the minimum of the i4 array *arr and returns it. */

  i4 i, location;
  i4 amin;
  /* initialize amin and location to 0th element */
  amin = *(arr + 0);
  location = 0;
  for (i = 1; i < xsize; i++)
    {
      if (*(arr + i) < amin)
	{
	  amin = *(arr + i);
	  location = i;
	}
    }
  return location;
}

i4 interpcc2d (double *fdata, double xmiss, i4 nx, i4 ny, 
    double *xwant, i4 nxinterp, double *ywant, i4 nyinterp, double **finterp)
{
  /*
   * This function does cubic convolution interpolation onto an array 
   * finterp from data defined in array fdata.  nx, ny are the
   * assumed dimensions of fdata, and nxinterp, nyinterp are the
   * assumed dimensions of finterp.  The values of x,y at which
   * the interpolation is desired are passed in through the arrays
   * xwant and ywant, which are dimensioned nxinterp and nyinterp,
   * respectively.  It is assumed that xwant, ywant are in units of
   * the indices of the original data array (fdata), 
   * treated as floating point (double precision, actually) 
   * numbers. Arrays fdata, xwant, and ywant are passed in
   * as pointers; The array finterp is defined in this function
   * as a "double" pointer and the array is created and passed back to
   * the calling function.  In the calling function, finterp is declared
   * as a pointer, but when it is passed into this function as
   * an argument, the address of the pointer is used.
   * 
   * if any of the datapoints within a kernel weighting distance of
   * xwant and ywant are equal to xmiss,
   * the returned value of finterp is also set to xmiss.  xmiss is a user-
   * defineable calling argument.
   */

  double *cdata;
/*  double txt, tyt, xint, yint, ftmp, xmiss = 0.; */
  double txt, tyt, xint, yint, ftmp;

  /* Logic for a user-defined value of xmiss has been added.  Previously
   * was just set to 0 as a local variable */

  double tx, ty, rx, ry;
  i4 i, ii, j, jj, itemp, jtemp, izero, jzero, databad;
/*  i4 transp; */

  /* Now, create the cdata array, bigger by 1 gp than fdata
   * all around the borders: */

  cdata = (double *) malloc (sizeof (double) * (nx + 2) * (ny + 2));

  /* Now fill the interior of cdata with fdata */

  for (i = 0; i < nx; i++)
    {
      for (j = 0; j < ny; j++)
	{
	  *(cdata + (i + 1)*(ny + 2) + (j + 1)) = *(fdata + i*ny + j);
	}
    }

  /*
   * The basic concept for filling in edges and corners of cdata is this:
   * The edge point is equal to 3*(value of adjacent point)
   * -3*value(next to adjacent point) + 1*value(3rd point).  This
   * prescription yields an extrapolation which is consistent with
   * a 3rd (or is it 4th?) order Taylor expansion of the function
   * evaluated at the last real gridpoint, and extrapolated to the
   * edge point.  This procedure is followed
   * thoughout here, though I think it isn't really correct for the
   * corner points because there I think an expansion from both
   * both directions should be done.  But no harm seems to be done
   * to the results.
   */

  /* Fill in the edges of cdata: */

  for (j = 0; j < ny; j++)
    {

      /* left and right edges: */

      *(cdata + 0*(ny + 2) + (j+1)) = *(fdata + 2*ny + j)
	- 3. * (*(fdata + 1*ny + j)) + 3. * (*(fdata + 0*ny + j));

      *(cdata + (nx + 1)*(ny + 2) + (j + 1)) = *(fdata + (nx - 3)*ny + j)
	- 3. * (*(fdata + (nx - 2)*ny + j)) + 3. * (*(fdata + (nx - 1)*ny + j));
    }
  for (i = 0; i < nx; i++)
    {

      /* bottom and top edges: */

      *(cdata + (i + 1)*(ny + 2) + 0) = *(fdata + i*ny + 2)
	- 3. * (*(fdata + i*ny + 1)) + 3. * (*(fdata + i*ny + 0));

      *(cdata + (i + 1)*(ny + 2) + ny + 1) = *(fdata + i*ny + ny - 3)
	- 3. * (*(fdata + i*ny + ny - 2)) + 3. * (*(fdata + i*ny + ny - 1));
    }

  /* Now fill in the 4 corners: */

  *(cdata + 0*(nx + 2) + 0) = 
    3. * (*(cdata + 1*(ny + 2) + 0)) -
    3. * (*(cdata + 2*(ny + 2) + 0)) + *(cdata + 3*(ny + 2) + 0);

  *(cdata + (nx + 1)*(ny + 2) + 0) = 
    3. * (*(cdata + nx*(ny + 2) + 0)) -
    3. * (*(cdata + (nx - 1)*(ny + 2) + 0)) + *(cdata +
						    (nx - 2)*(ny + 2) + 0);

  *(cdata + 0*(ny + 2) + ny + 1) = 
    3. * (*(cdata + 0*(ny + 2) + ny)) -
    3. * (*(cdata + 0*(ny + 2) + ny - 1)) + *(cdata + 0*(ny + 2) + ny - 2);

  *(cdata + (nx + 1)*(ny + 2) + ny + 1) =
    3. * (*(cdata + nx*(ny + 2) + ny + 1)) -
    3. * (*(cdata + (nx - 1)*(ny + 2) + ny + 1)) + *(cdata +
						       (nx - 2)*(ny + 2) +
						       ny + 1);

  /* Now create the space for finterp */

  *finterp = (double *) malloc (sizeof (double) * nxinterp * nyinterp);

  /* Now interpolate onto the desired grid */

  for (i = 0; i < nxinterp; i++)
    {
      /* starting the outer loop over x */

      xint = *(xwant + i);

      /* make sure izero is in bounds */

      itemp = ((i4) xint > 0) ? (i4) xint : 0;
      izero = (itemp < (nx - 2)) ? itemp : nx - 2;
      for (j = 0; j < nyinterp; j++)
	{
	  /* starting the outer loop over y */

	  yint = *(ywant + j);
	  if ((yint < 0.) || (yint > (double) (ny - 1))
	      || ((xint < 0) || (xint > (double) (nx - 1))))
	    {
	      /* if data missing, set interp to xmiss */

/* Debug
              printf("interpccd2: i=%d,j=%d gets finterp[i,j] set to xmiss\n",
                    i,j);
*/
	      *(*finterp + i * nyinterp + j) = xmiss;
	    }
	  else
	    {
	      /* make sure jzero is in bounds */

	      jtemp = ((i4) yint > 0) ? (i4) yint : 0;
	      jzero = (jtemp < (ny - 2)) ? jtemp : ny - 2;

	      /* initialize the temporary finterp value */

	      ftmp = (double) 0.;

	      /* start the innermost loops over neighboring
	       * data points*/

              databad=0;
	      for (ii = -1; ii < 3; ii++)
		{
		  txt = xint - (double) (izero + ii);
		  tx = (double) fabs (txt);

		  /* evaluate kernel wt function r(tx): */

		  /* Note no testing for out of range 
		   * values of |tx| or |ty| > 2 --
		   * we assume the tx, ty are properly
		   * computed such that their absolute
		   * value never exceeds 2. */

		  rx = (tx >= (double) 1.0) ?
		    (((((double) (-0.5)) * tx +
		       ((double) 2.5)) * tx) -
		     (double) 4.) * tx + (double) 2. :
		    (((double) (1.5)) * tx -
		     ((double) (2.5))) * tx * tx + (double) 1.;

		  for (jj = -1; jj < 3; jj++)
		    {

		      tyt = yint - (double) (jzero + jj);
		      ty = (double) fabs (tyt);

		      /* evaluate kernel weighting
		       * function r(ty): */

		      ry = (ty >= (double) 1.0) ?
			(((((double) (-0.5)) * ty +
			   ((double) 2.5)) * ty) -
			 (double) 4.) * ty + (double) 2. :
			(((double) (1.5)) * ty -
			 ((double) (2.5))) * ty * ty + (double) 1.;

		      /* do the cubic convolution
		       * over the neighboring data
		       * points, using the x and
		       * y evaluated kernel weighting
		       * functions rx and ry: */

		      ftmp = ftmp +
			*(cdata + (izero + 1 + ii)*(ny + 2)
			  + jzero + 1 + jj) * rx*ry;
                      if( *(cdata+(izero+1+ii)*(ny+2)+jzero+1+jj) == xmiss)
                          databad=1;
		    }
		}
	      /* now assign this value to interpolated
	         array, unless one of the values was xmiss: */
              if(databad)
              {
/* Debug
                 printf("interpcc2d: i=%d,j=%d gives databad\n",i,j);
*/
                 *(*finterp + i*nyinterp + j) = xmiss;
              }
              else
              {
	         *(*finterp + i*nyinterp + j) = ftmp;
              }
	    }
	}
    }


/* DEBUG
  transp=1;
  writeimage("cdata.dat",cdata,nx+2,ny+2,transp);
*/

  /* free the cdata space */
  free (cdata);

  /* we're done */

  return 0;
}

i4 byteswapflct (unsigned char *arr, i4 arrsize, i4 nbpw)
/* Pretty simple:  arr is input array, which is byte-swapped in place,
   nbpw is the number of bytes per word, and arrsize is the size of the array
   (in units of nbpw bytes).  It is assumed that arr has
   already have been correctly defined and allocated in the calling program. */
{
  i4 i, j;
  unsigned char temp;
  for (i = 0; i < arrsize; i++)	/* the loop over the array elements */
    {
      for (j = 0; j < nbpw/2; j++)/* the loop over bytes in a single element */
	{
	  temp = *(arr + i*nbpw + (nbpw - j - 1));
	  *(arr + i*nbpw + (nbpw - j - 1)) = *(arr + i*nbpw + j);
	  *(arr + i*nbpw + j) = temp;
	}
    }
  return 0;
}

i4 is_large_endian ()
/* This function returns 1 if it is a large endian machine, 0 otherwise */
{
  const unsigned char fakeword[4] = { 0xFF, 0x00, 0xFF, 0x00 };
  i4 realword = *((i4 *) fakeword);
  if (realword == 0xFF00FF00)
    {
      return 1;
    }
  else
    {
      return 0;
    }
}

double signum(double v)
{
/* signum function in C since it's apparently not in math.h */
  if (v < 0) return -1.0;
  if (v > 0) return 1.0;
  return 0.0;
}

i4 pc2mc(i4 transp, double *f, i4 nx, i4 ny, double lonmin, double lonmax, 
         double latmin, double latmax, double ** finterp, i4 *nxinterp, 
         i4 *nyinterp, double * vmin, double *vmax)
{
/* interpolate input array f, in Plate Carree coordinates, to output
array *finterp, in Mercator projected coordinates.  Uses math.h 

If (transp) then longitude index varies fastest, corresponding to
the "y" direction in interpcc2d.  So if (transp), latitude index varies 
more slowly, and is associated with "x" direction here:

In notation below, the "u" direction in Mercator projection is in the
azimuthal direction, and the "v" direction is in the latitude direction.

*/

/* local variable definitions */

double umin,umax,dellon,dellat,delu,delv;
double dlonm1,dlatm1,expfac;
double xmiss=-999999.;
double *vrange=NULL,*latwant=NULL,*xwant=NULL,*ywant=NULL;
double *lonwant=NULL;
i4 i,nvinterp,nuinterp,ier;

/* first check that latmin < latmax, lonmin < lonmax: */

if(latmax <= latmin)
{
   printf ("pc2mc: latmax <= latmin, error\n");
   exit(1);
}
if(lonmax <= lonmin)
{
   printf ("pc2mc: lonmax <= lonmin, error\n");
   exit(1);
}

/* First get dellon, dellat from lon,lat range */

if(transp)
{
  dellon=(lonmax-lonmin)/((double)(ny-1));
  dellat=(latmax-latmin)/((double)(nx-1));
}
else
{
  dellon=(lonmax-lonmin)/((double)(nx-1));
  dellat=(latmax-latmin)/((double)(ny-1));
}
dlonm1=1./dellon;
dlatm1=1./dellat;

/* Compute umin,umax: */
umin=lonmin;
umax=lonmax;

/* Compute vmin,vmax: */
*vmin=signum(latmin)*log((1.0+sin(fabs(latmin)))/cos(fabs(latmin)));
*vmax=signum(latmax)*log((1.0+sin(fabs(latmax)))/cos(fabs(latmax)));

/* Compute array index range in x,y directions for *finterp.

First calculate expansion factor in lat direction, then nvinterp, delv, 
rangey.  The objective here is to make sure there's enough resolution in
the v direction so that the original array is not undersampled. */

expfac=(*vmax-*vmin)/(latmax-latmin);

if (transp)
{
/* in this case, latitude is in the slowly varying index direction */
/*  nvinterp=(i4) nx*expfac + 1; */
  nvinterp=(i4) nx*expfac;
  *nxinterp=nvinterp;
  delv=(*vmax-*vmin)/(nvinterp-1);
/* in lon direction, no expansion */
  nuinterp=ny;
  *nyinterp=nuinterp;
  delu=(umax-umin)/(nuinterp-1);
}
else
{
/* in this case, latitude is in the rapidly varying index direction */
/* nvinterp=(i4) ny*expfac + 1; */
  nvinterp=(i4) ny*expfac; 
  *nyinterp=nvinterp;
  delv=(*vmax-*vmin)/(nvinterp-1);
/* in lon direction, no expansion */
  nuinterp=nx;
  *nxinterp=nuinterp;
  delu=(umax-umin)/(nuinterp-1);
}

/* Compute uniformly spaced array vrange and corresponding latwant for Mercator 
  projection */

vrange=malloc(nvinterp*sizeof(double));
latwant=malloc(nvinterp*sizeof(double));
for(i=0;i<nvinterp;i++)
{
/* Compute array of desired latitudes corresponding to vrange array.
   Note this makes use of the relationship sin(lat[i])=tanh(vrange[i]) */
   vrange[i]=*vmin + i*delv;
   latwant[i]=asin(tanh(vrange[i]));
}
/* Now compute urange for Mercator projection */
lonwant=malloc(nuinterp*sizeof(double));
for (i=0;i<nuinterp;i++)
{
   lonwant[i]=lonmin + i*delu;
}

/* define xwant, ywant for interpcc2d */

if(transp)
{
  xwant=malloc(nvinterp*sizeof(double));
  ywant=malloc(nuinterp*sizeof(double));
  for (i=0;i<nvinterp;i++)
  {
      xwant[i]=dlatm1*(latwant[i]-latmin);
  }
  for (i=0;i<nuinterp;i++)
  {
      ywant[i]=dlonm1*(lonwant[i]-lonmin);
  }
}
else
{
  ywant=malloc(nvinterp*sizeof(double));
  xwant=malloc(nuinterp*sizeof(double));
  for (i=0;i<nvinterp;i++)
  {
     ywant[i]=dlatm1*(latwant[i]-latmin);
  }
  for (i=0;i<nuinterp;i++)
  {
     xwant[i]=dlonm1*(lonwant[i]-lonmin);
  }
}

/* add here a test for roundoff errors making endpoints of xwant,ywant
extend beyond the limits 0, nx-1, 0, ny-1 */

if(xwant[0] < 0.) xwant[0]=0.;
if(xwant[*nxinterp-1] > (double) (nx-1)) xwant[*nxinterp-1]=
  (double) (nx-1);
if(ywant[0] < 0.) ywant[0]=0.;
if(ywant[*nyinterp-1] > (double) (ny-1)) ywant[*nyinterp-1]=
  (double) (ny-1);

ier=interpcc2d(f,xmiss,nx,ny,xwant,*nxinterp,ywant,*nyinterp,finterp);

/* Here output various diagnostics for debugging */

/* free all the local allocated arrays */

free(xwant);
free(ywant);
free(lonwant);
free(vrange);
free(latwant);

/* In calling program, might want to test if any values of finterp are 
equal to xmiss, and if so, set them to 0? or maybe do that here?*/

return 0;
}

i4 mc2pc(i4 transp, double *f, i4 nxinterp, i4 nyinterp, double umin, 
         double umax, double vmin, double vmax, double ** finterp, i4 nx, 
         i4 ny)
{
/* interpolate input array f, in Mercator coordinates, to output
array *finterp, in Plate Carree projected coordinates.  Uses math.h 

If (transp) then longitude index varies fastest, corresponding to
the "y" direction in interpcc2d.  So if (transp), latitude index varies 
more slowly, and is associated with "x" direction here.

In notation below, the "u" direction in Mercator projection is in the
azimuthal direction, and the "v" direction is in the latitude direction.

Note some differences here from mc2pc and pc2mc:  While in pc2mc, nxinterp
and nyinterp are computed within the function and returned as output variables,
here nxinterp, nyinterp, nx, and ny are all specified as input variables.  
This is to ensure that when going back from Mercator to (lon,lat) 
that nx,ny are the same as what you start with.  There shouldn't need to be
an expansion factor going in this direction that was needed in going from
pc to mc. */

/* local variable definitions */

double lonmin,lonmax,latmin,latmax,dellon,dellat,delu,delv;
double dum1,dvm1;
double xmiss=-999999.;
double *latrange=NULL,*vwant=NULL,*xwant=NULL,*ywant=NULL;
double *lonrange=NULL,*latwant=NULL,*uwant=NULL;
i4 i,nv,nu,nvinterp,nuinterp,ier;

/* first check that vmin < vmax, umin < umax: */

if(vmax <= vmin)
{
   printf ("mc2pc: vmax <= vmin, error\n");
   exit(1);
}
if(umax <= umin)
{
   printf ("mc2pc: umax <= umin, error\n");
   exit(1);
}

/* First get delu, delv from u,v range */

if(transp)
{
  delu=(umax-umin)/((double)(nyinterp-1));
  delv=(vmax-vmin)/((double)(nxinterp-1));
}
else
{
  delu=(umax-umin)/((double)(nxinterp-1));
  delv=(vmax-vmin)/((double)(nyinterp-1));
}
dum1=1./delu;
dvm1=1./delv;

/* Compute lonmin,lonmax: */
lonmin=umin;
lonmax=umax;

/* Compute latmin,latmax: */

latmin=asin(tanh(vmin));
latmax=asin(tanh(vmax));

if (transp)
{
/* in this case, latitude is in the slowly varying index direction */
  nv=nx;
  nvinterp=nxinterp;
  dellat=(latmax-latmin)/(nv-1);
/* in lon direction */
  nu=ny;
  nuinterp=nyinterp;
  dellon=(lonmax-lonmin)/(nu-1);
}
else
{
/* in this case, latitude is in the rapidly varying index direction */
  nv=ny;
  nvinterp=nyinterp;
  dellat=(latmax-latmin)/(nv-1);
/* in lon direction */
  nu=nx;
  nuinterp=nxinterp;
  dellon=(lonmax-lonmin)/(nu-1);
}

/* Compute uniformly spaced array latrange and corresponding vwant for 
Plate Carree projection */

latrange=malloc(nv*sizeof(double));
vwant=malloc(nv*sizeof(double));
for(i=0;i<nv;i++)
{
/* Compute array of desired v corresponding to latrange array.
   Note this makes use of the relationship v=log((1.+sin[lat])/cos(lat)) */

   latrange[i]=latmin + i*dellat;
   vwant[i]=signum(latrange[i]) * log( ( (1.+sin(fabs(latrange[i]))) ) /
            cos(fabs(latrange[i])) );
}
/* Now compute lon range for Plate Carree projection */
lonrange=malloc(nu*sizeof(double));
uwant=malloc(nu*sizeof(double));
for (i=0;i<nu;i++)
{
   lonrange[i] = lonmin + i*dellon;
   uwant[i]= umin + i*(umax-umin)/(nu-1);
}

/* define xwant, ywant for interpcc2d */

if(transp)
{
  xwant=malloc(nv*sizeof(double));
  ywant=malloc(nu*sizeof(double));
  for (i=0;i<nv;i++)
  {
      xwant[i]=dvm1*(vwant[i]-vmin);
  }
  for (i=0;i<nu;i++)
  {
      ywant[i]=dum1*(uwant[i]-umin);
  }
}
else
{
  ywant=malloc(nv*sizeof(double));
  xwant=malloc(nu*sizeof(double));
  for (i=0;i<nv;i++)
  {
     ywant[i]=dvm1*(vwant[i]-vmin);
  }
  for (i=0;i<nu;i++)
  {
     xwant[i]=dum1*(uwant[i]-umin);
  }
}

/* add here a test for roundoff errors making endpoints of xwant,ywant
extend beyond the limits 0, nxinterp-1, 0, nyinterp-1 */

if(xwant[0] < 0.) xwant[0]=0.;
if(xwant[nx-1] > (double) (nxinterp-1)) xwant[nx-1]=
  (double) (nxinterp-1);
if(ywant[0] < 0.) ywant[0]=0.;
if(ywant[ny-1] > (double) (nyinterp-1)) ywant[ny-1]=
  (double) (nyinterp-1);

ier=interpcc2d(f,xmiss,nxinterp,nyinterp,xwant,nx,ywant,ny,finterp);

/* Here output various diagnostics for debugging */

/* free all allocated arrays */

free(xwant);
free(ywant);
free(uwant);
free(vwant);
free(latwant);
free(latrange);
free(lonrange);

return 0;
}

i4 flct_pc (i4 transp, double * f1, double * f2, i4 nx, i4 ny, double deltat, 
    double deltas, double sigma, double * vx, double * vy, double * vm,
    double thresh, i4 absflag, i4 filter, double kr, i4 skip,
    i4 poffset, i4 qoffset, i4 interpolate, double latmin, double latmax, 
    i4 biascor, i4 verbose) 
{

/* BEGIN FLCT_PC FUNCTION */
/* The idea here is to interpolate two magnetograms f1,f2 to Mercator
   projected coordinates, then run FLCT on those two images, and then
   interpolate those velocity results back to Plate Carree coordinates.  Once
   this is done, both velocity components are modulated by cos(latitude). */

/*  char *version ="1.04    "; */

i4 nxinterp,nyinterp,nxf,nyf,nu,nv,i,j,ier,absflag0,iloc1,iloc2,belowthresh;
i4 sigmaeq0;
double thresh0, f1max, f2max, fmax, lonmin, lonmax, dellat, dellon, vmin, vmax,
       fabsbar, latbar;
double *f1merc=NULL,*f2merc=NULL,*vxmerc=NULL,*vymerc=NULL,*vmmerc=NULL;
double *vxint=NULL,*vyint=NULL;
double *f1temp=NULL,*f2temp=NULL;
double *latrange=NULL;

/* Note that at this point, nx,ny do not necessarily correspond to the
slowly varying and rapidly varying index directions, resp.  Must be careful, 
because in pc2mc and mc2pc ny is assumed to be rapidly varying, nx slowly 
varying. 

Approach here is to do what is done in flct function itself, and flip the
roles of nx,ny (if transp) to mirror what is assumed in flct.  Still 
retain original nx,ny in case they're needed.
So:  Will define variables nxf,nyf that are flipped from nx,ny if(transp). 
Note flct is called with nxinterp,nyinterp.

If sigma == 0, will set nx,ny to 1.
*/


if(transp)
{
   nxf=ny;
   nyf=nx;
}
else
{
   nxf=nx;
   nyf=ny;
}

sigmaeq0=0;
if(sigma == (double)0) sigmaeq0=1;

if(sigmaeq0)
{
   nx=1;
   ny=1;
}

/* Note that in flct_pc, we can't use skip without interpolate: */

if((skip != 0) && (interpolate == 0))
{
  printf("flct_pc:  If using skip, must also use interpolate. Exiting\n");
  exit(0);
}

/* Now associate latitude and longitude directions with nxf,nyf, depending
on what transp is */

if(transp)
{
   nv=nxf;
   nu=nyf;
}
else
{
  nv=nyf;
  nu=nxf;
}

dellat=(latmax-latmin)/(nv-1);

/* Use the fact that in HMI Plate Carree, dellat and dellon are equal: */

dellon=dellat;
lonmin=0.;
lonmax=(nu-1)*dellon;

/* compute 1D array of latitudes: */

latrange=malloc(nv*sizeof(double));
latbar=0.;
for(i=0;i<nv;i++)
{
  latrange[i]=latmin+i*dellat; 
  latbar+=latrange[i];
}

/* Compute average latitude latbar */

latbar/=(double)nv;

/* Interpolate f1, f2 images to Mercator coordinates, get nxinterp, nyinterp
   vmin, vmax: */

ier=pc2mc(transp,f1,nxf,nyf,lonmin,lonmax,latmin,latmax,&f1merc,&nxinterp,
    &nyinterp,&vmin,&vmax);
ier=pc2mc(transp,f2,nxf,nyf,lonmin,lonmax,latmin,latmax,&f2merc,&nxinterp,
    &nyinterp,&vmin,&vmax);

/*
printf("nxf=%d, nyf=%d, nxinterp=%d,nyinterp=%d\n",nxf,nyf,nxinterp,nyinterp);
*/

/* Allocate memory for vxmerc, vymerc, vmmerc before FLCT call */

if(!sigmaeq0)
{
  vxmerc=malloc(nxinterp*nyinterp*sizeof(double));
  vymerc=malloc(nxinterp*nyinterp*sizeof(double));
  vmmerc=malloc(nxinterp*nyinterp*sizeof(double));
}
else
{
  vxmerc=malloc(1*sizeof(double));
  vymerc=malloc(1*sizeof(double));
  vmmerc=malloc(1*sizeof(double));
}

/* Because we don't want to interpolate computed points with points below
threshold, make sure we compute all points with flct and then set points to
0 after we return to Plate Carree coordinates.  Accomplish this by calling
flct with thresh=0 and absflag=0: */

thresh0=0.;
absflag0=0;

/* Now call the FLCT function, doing LCT on f1merc and f2merc: */

/* Need to flip order of nxinterp, nyinterp if transp */

if(transp)
{
    if(verbose) 
    {
      printf("flct_pc: No. of points in v - Mercator Proj: %d\n", nxinterp);
      fflush(stdout);
    }
    if(verbose) 
    {
      printf("flct_pc: effective value of sigma in Merc Proj: %g\n",
      sigma * (double) nxf / (double) nxinterp );  
      fflush(stdout);
    }
    ier=flct(transp,f1merc,f2merc,nyinterp,nxinterp,deltat,deltas,sigma,
    vxmerc,vymerc,vmmerc,thresh0,absflag0,filter,kr,skip,poffset,qoffset,
    interpolate,biascor,verbose);
}
else
{
    if(verbose) 
    {
      printf("flct_pc: No. of points in v - Mercator Proj: %d\n",
      nyinterp);
      fflush(stdout);
    }
    if(verbose) 
    {
      printf("flct_pc: effective value of sigma in Merc Proj: %g\n",
      sigma * (double) nyf / (double) nyinterp );  
      fflush(stdout);
    }
    ier=flct(transp,f1merc,f2merc,nxinterp,nyinterp,deltat,deltas,sigma,
    vxmerc,vymerc,vmmerc,thresh0,absflag0,filter,kr,skip,poffset,qoffset,
    interpolate,biascor,verbose);
}

/* Now interpolate vxmerc, vymerc back to Plate Carree grid (vxint,vyint): */
if(!sigmaeq0)
{
  ier=mc2pc(transp,vxmerc,nxinterp,nyinterp,lonmin,lonmax,vmin,vmax,&vxint,
    nxf,nyf);
  ier=mc2pc(transp,vymerc,nxinterp,nyinterp,lonmin,lonmax,vmin,vmax,&vyint,
    nxf,nyf);
}

/* Next, modulate velocities by cos(latitude): */

if(!sigmaeq0)
  {

  if(transp) /*  Column major case (latitudes vary slowest) */
  {
     for(i=0;i<nxf;i++)
        {
        for(j=0;j<nyf;j++)
           {
             vx[i*nyf+j]=cos(latrange[i])*vxint[i*nyf+j];
             vy[i*nyf+j]=cos(latrange[i])*vyint[i*nyf+j];
             vm[i*nyf+j]=0.5; /* all values interpolated, so vm=0.5 */
           }
        }
  }
  else /* Row major case (latitudes vary fastest) */
  {
     for(i=0;i<nxf;i++)
        {
        for (j=0;j<nyf;j++)
            {
              vx[i*nyf+j]=cos(latrange[j])*vxint[i*nyf+j];
              vy[i*nyf+j]=cos(latrange[j])*vyint[i*nyf+j];
              vm[i*nyf+j]=0.5; /* all values interpolated, so vm=0.5 */
            }
        }
  }
}

/* Now put in logic to set points below threshold to 0: (this code copied from
   flct function itself, since we have to repeat its logic) */
if ((thresh > 0.) && (thresh < 1.) && (absflag == 0) && (!sigmaeq0))

{
  f1temp = (double *) malloc (sizeof (double) * nxf * nyf);
  f2temp = (double *) malloc (sizeof (double) * nxf * nyf);

  for (i = 0; i < nxf * nyf; i++)

  {

    /* compute abs value of f1,f2 arrays as f1temp,
     * f2temp arrays */

    *(f1temp + i) = (double) fabs (*(f1 + i));
    *(f2temp + i) = (double) fabs (*(f2 + i));
  }
  /* now find maximum absolute value of both images */

  iloc1 = maxloc (f1temp, nxf * nyf);
  iloc2 = maxloc (f2temp, nxf * nyf);
  f1max = *(f1temp + iloc1);
  f2max = *(f2temp + iloc2);
  fmax = (f1max > f2max) ? f1max : f2max;

  /* now convert relative thresh to absolute threshhold */

  thresh *= fmax;
  if (verbose) 
  {
    printf ("flct_pc: relative threshold in abs. units = %g\n", thresh);
    fflush(stdout);
  }

  free (f1temp);
  free (f2temp);
}

/* Now set vx,vy,vm to 0 if image points are below threshold: */

if(!sigmaeq0)
{
for (i = 0; i < nxf; i++)
    {
      for (j = 0; j < nyf; j++)
        {
            fabsbar = 0.5 * (fabs (*(f1 + i * nyf + j) + *(f2 + i * nyf + j)));
            belowthresh = (fabsbar < thresh);
            if (belowthresh)
            {
               *(vx+i*nyf+j)=0.;
               *(vy+i*nyf+j)=0.;
               *(vm+i*nyf+j)=0.;
            }
        }
    }
}
if(sigmaeq0) /* Only a single value is computed if sigma == 0 */
{
  vx[0] = cos(latbar) * vxmerc[0];
  vy[0] = cos(latbar) * vymerc[0];
  vm[0] = 0.5; /* in this case we should say vx,vy interpolated */
  if(verbose) 
  {
     printf("flct_pc: After cos(latbar) modulation, vx = %g, vy =%g\n" 
       ,vx[0],vy[0]);
     fflush(stdout);
  }
}

/* debugging code block:  write out f1merc,f2merc,vxmerc,vymerc,vxint,vyint */
{
  i4 debug=0;
  if(debug)
  {
    ier=write2images("f1f2merc.dat",f1merc,f2merc,nxinterp,nyinterp,transp);
    ier=write2images("vxvymerc.dat",vxmerc,vymerc,nxinterp,nyinterp,transp);
    ier=write2images("vxintvyint.dat",vxint,vyint,nxf,nyf,transp);
  }

}

/* we're done! */

free(latrange);
free(f1merc);
free(f2merc);
free(vxmerc);
free(vymerc);
free(vmmerc);
free(vxint);
free(vyint);
return 0;
/*  END FLCT_PC FUNCTION */
}

void flct_pc_f77__(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny, 
     double * deltat, double * deltas, double * sigma, double * vx, 
     double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter, 
     double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
     double * latmin, double * latmax, i4 * biascor, i4 * verbose)

{
i4 ierflct;

/* Now call the C version of the flct function, dereferencing all the
   variables that aren't pointers within the C flct function */

ierflct=flct_pc(*transp,f1,f2,*nx,*ny,*deltat,*deltas,*sigma,vx,vy,vm,
     *thresh,*absflag,*filter,*kr,*skip,*poffset,*qoffset,*interpolate,
     *latmin, *latmax, *biascor, *verbose);

return;
}

void flct_pc_f77_(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny, 
     double * deltat, double * deltas, double * sigma, double * vx, 
     double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter, 
     double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
     double * latmin, double * latmax, i4 * biascor, i4 * verbose)

{
i4 ierflct;

/* Now call the C version of the flct_pc function, dereferencing all the
   variables that aren't pointers within the C flct function */

ierflct=flct_pc(*transp,f1,f2,*nx,*ny,*deltat,*deltas,*sigma,vx,vy,vm,
     *thresh,*absflag,*filter,*kr,*skip,*poffset,*qoffset,*interpolate,
     *latmin, *latmax, *biascor, *verbose);

return;
}

void flct_pc_f77(i4 * transp, double * f1, double * f2, i4 * nx, i4 * ny, 
     double * deltat, double * deltas, double * sigma, double * vx, 
     double * vy, double * vm, double * thresh, i4 * absflag, i4 * filter, 
     double * kr, i4 * skip, i4 * poffset, i4 * qoffset, i4 * interpolate,
     double * latmin, double * latmax, i4 * biascor, i4 * verbose)

{
i4 ierflct;

/* Now call the C version of the flct_pc function, dereferencing all the
   variables that aren't pointers within the C flct function */

ierflct=flct_pc(*transp,f1,f2,*nx,*ny,*deltat,*deltas,*sigma,vx,vy,vm,
     *thresh,*absflag,*filter,*kr,*skip,*poffset,*qoffset,*interpolate,
     *latmin, *latmax, *biascor, *verbose);

return;
}

i4 shift_frac2d(double *arr, double delx, double dely, double *outarr,
        i4 nx, i4 ny, i4 transp, i4 verbose)

{

/* uncomment next line if to be compiled by itself */
/* #include <fftw3.h> */

  i4 i,j, nxtmp, nytmp;
  double *kx, *ky, *fftdeltrx, *fftdeltry,*fftdeltix,*fftdeltiy;
  double normfac, dxarg, dyarg,fftdeltr,fftdelti,outar,outai;
  double pi=3.1415926535898;
  fftw_complex *ina, *outa;
  fftw_plan pa, pbacka;

  /* if verbose=0, no output to stdout, if verbose=1, send msgs to stdout */

  /* flip nx, ny if (transp) */

  if(transp)
  {
     nxtmp=nx;
     nytmp=ny;
     nx=nytmp;
     ny=nxtmp;
  }
  /* allocate memory for FFT arrays and intermediate arrays */
  outa = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) *
     nx * ny );
  ina = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) * nx * ny);
  kx=(double *) fftw_malloc (sizeof(double)*nx);
  fftdeltrx=(double *) fftw_malloc (sizeof(double)*nx);
  fftdeltix=(double *) fftw_malloc (sizeof(double)*nx);
  ky=(double *) fftw_malloc (sizeof(double)*ny);
  fftdeltry=(double *) fftw_malloc (sizeof(double)*ny);
  fftdeltiy=(double *) fftw_malloc (sizeof(double)*ny);
  /* set up plans for FFTs: */
  pa = fftw_plan_dft_2d (nx, ny, ina, outa, FFTW_FORWARD, FFTW_MEASURE);
  pbacka = fftw_plan_dft_2d (nx, ny,outa, ina, FFTW_BACKWARD, FFTW_MEASURE);
  /* compute spatial frequencies kx and ky */
  make_freq(kx,nx);
  make_freq(ky,ny);
 /* calculate normalization factor */
  normfac = (1. / ((double) nx * ny));
 /* assign image fft array (complex) to input data array (double) */
      for (i = 0; i < nx * ny; i++)
	{
#ifdef COMPLEXH
          /* if complex.h included, do this: */
          ina[i]=arr[i]+I*0.;
#else
          /* If complex.h not included, do this: */
	  ina[i][0] = arr[i]; /* real part */
	  ina[i][1] = (double) 0.; /* imag part */
#endif

	}

  /* do the forward FFT: */
  
  fftw_execute (pa);
  
  /* get shifts normalized to nx, ny.  Note if transp==1 then roles of delx,
  dely are switched.  This occurs if array is column-major.  */

  if (transp)
  {
     dxarg= -dely/((double) nx);
     dyarg= -delx/((double) ny);
  }
  else
  {
     dxarg= -delx/((double) nx);
     dyarg= -dely/((double) ny);
  }

  /* calculate FFT of delta function at delx, dely, then mpy by FFT of image */

    /* calculate sin, cosine factors first */
    for (i=0;i<nx;i++)
    {
        fftdeltrx[i]=cos(2.*pi*kx[i]*dxarg);
        fftdeltix[i]=sin(2.*pi*kx[i]*dxarg);
    }
    for (j=0;j<ny;j++)
    {
        fftdeltry[j]=cos(2.*pi*ky[j]*dyarg);
        fftdeltiy[j]=sin(2.*pi*ky[j]*dyarg);
    }
    /* now compute fft of shifted image */
    for (i=0;i<nx;i++)
    {
      for (j=0;j<ny;j++)
      {
         /* real part of exp(i kx dxarg + i ky dyarg) */
         fftdeltr=fftdeltrx[i]*fftdeltry[j]-fftdeltix[i]*fftdeltiy[j];
         /* imag part of exp(i kx dxarg + i ky dyarg) */
         fftdelti=fftdeltrx[i]*fftdeltiy[j]+fftdeltix[i]*fftdeltry[j];

#ifdef COMPLEXH
         /* If complex.h included, do this: */
         outar=creal(outa[i*ny+j]);
         outai=cimag(outa[i*ny+j]);
#else
         /* If complex.h not included, do this: */
         outar=outa[i*ny+j][0];
         outai=outa[i*ny+j][1];
#endif

#ifdef COMPLEXH
         /* If complex.h included, do this: */
         outa[i*ny+j]=outar*fftdeltr-outai*fftdelti
                    + I*(outar*fftdelti+outai*fftdeltr);
#else
         /* If complex.h not included, do this: */
         /* real part of fft of shifted fn */
         outa[i*ny+j][0]=(outar*fftdeltr-outai*fftdelti);
         /* imag part of fft of shifted fn */
         outa[i*ny+j][1]=(outar*fftdelti+outai*fftdeltr);
#endif
      }
    }

   /* Do inverse FFT to get shifted image fn back into the ina complex array */

    fftw_execute (pbacka);

   /* Output array is the real part of ina (imag. part should be zero) */

  for (i = 0; i < nx * ny; i++)
    {
#ifdef COMPLEXH
      /* If complex.h included do this: */
      outarr[i]=creal(ina[i])*normfac;
#else
      /* If complex.h not included, do this: */
      outarr[i]=(double) (ina[i][0])*normfac;
#endif

    }

/* Free the plans and locally created arrays */

      fftw_free (outa);
      fftw_free (kx);
      fftw_free (ky);
      fftw_free (fftdeltrx);
      fftw_free (fftdeltix);
      fftw_free (fftdeltry);
      fftw_free (fftdeltiy);
      fftw_free (ina);
      fftw_destroy_plan (pa);
      fftw_destroy_plan (pbacka);

/* done */

return 0;

}

i4 warp_frac2d(double *arr, double *delx, double *dely, double *outarr,
        i4 nx, i4 ny, i4 transp, i4 verbose)

{

/* uncomment if this function will be compiled on its own */
/* #include <fftw3.h> */

  i4 i,j,ii,jj,nxtmp,nytmp;
  double *kx, *ky, *snkx, *cskx, *snky, *csky;
  double pi=3.1415926535898;
  double normfac, fftdeltr, fftdelti,outar,outai,totalimij,nxinv,nyinv,
         xarg,yarg,shiftx,shifty;
  fftw_complex *ina, *outa; 
  fftw_plan pa;
  /* if verbose==0, no output to stdout, if verbose==1, msgs sent to stdout */
  /* flip nx, ny if (transp) */
  if(transp)
  {
     nxtmp=nx;
     nytmp=ny;
     nx=nytmp;
     ny=nxtmp;
  }
  /* allocate FFT input, output arrays, plus temporary arrays */
  outa = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) *
     nx * ny);
  ina = (fftw_complex *) fftw_malloc (sizeof (fftw_complex) * nx * ny);
  kx=(double *) fftw_malloc (sizeof(double)*nx);
  ky=(double *) fftw_malloc (sizeof(double)*ny);
  cskx=(double *) fftw_malloc (sizeof(double)*nx);
  snkx=(double *) fftw_malloc (sizeof(double)*nx);
  csky=(double *) fftw_malloc (sizeof(double)*ny);
  snky=(double *) fftw_malloc (sizeof(double)*ny);
  /* set up plan for forward FFT: */
  pa = fftw_plan_dft_2d (nx, ny, ina, outa, FFTW_FORWARD, FFTW_MEASURE);
  /* calculate spatial frequencies */
  make_freq(kx,nx);
  make_freq(ky,ny);
  /* calculate normalization factor */
  normfac = (1. / ((double) nx * ny));
  nxinv = 1./((double) nx);
  nyinv = 1./((double) ny);
  /* copy input array to real part of input fft array */
  for (i = 0; i < nx * ny; i++)
  {
#ifdef COMPLEXH
    /* Do this if complex.h included: */
    ina[i]=(double) arr[i] + I* (double) 0;
#else
    /* Do this if complex.h not included */
    ina[i][0] = (double) arr[i]; /* real part */
    ina[i][1] = (double) 0.; /* imag part */
#endif

  }

  /* actually do the forward FFT: */

  fftw_execute (pa);

  /* outer loop over spatial locations */
    for (i=0;i<nx;i++)
    {
      if(verbose)
      {
         printf ("warp: progress  i = %d out of %d\r", i, nx - 1);
         fflush(stdout);
      }
      for (j=0;j<ny;j++)
      {
         /* Note that if (transp) then array is assumed column major, switch
            roles of delx, dely */
         if(transp)
         {
            shiftx=dely[i*ny+j];
            shifty=delx[i*ny+j];
         }
         else
         {
            shiftx=delx[i*ny+j];
            shifty=dely[i*ny+j];
         }
         xarg=2.*pi*(((double) i) - shiftx)*nxinv;
         yarg=2.*pi*(((double) j) - shifty)*nyinv;
         for (ii=0;ii<nx;ii++)
         {
             /* compute kx-dependent sin, cos terms */
             cskx[ii]=cos(kx[ii]*xarg);
             snkx[ii]=sin(kx[ii]*xarg);
         }
         for (jj=0;jj<ny;jj++)
         {
             /* compute ky-dependent sin, cos terms */
             csky[jj]=cos(ky[jj]*yarg);
             snky[jj]=sin(ky[jj]*yarg);
         }

         /* initialize the integral over wavenumber */
         totalimij=(double) 0.;

         /* inner loop to integrate over kx, ky wavenumbers */
         for (ii=0;ii<nx;ii++)
             {
               for (jj=0;jj<ny;jj++)
                 {
                    /* compute real, im parts of exp (i [kx xarg + ky yarg]) */
                    fftdeltr=cskx[ii]*csky[jj]-snkx[ii]*snky[jj];
                    fftdelti=cskx[ii]*snky[jj]+snkx[ii]*csky[jj];
                    /* extract real, imag parts of fft of original image */

#ifdef COMPLEXH
                    /* If complex.h included, do this: */
                    outar=creal(outa[ii*ny+jj]);
                    outai=cimag(outa[ii*ny+jj]);
#else
                    /* If complex.h not included, do this: */
                    outar=outa[ii*ny+jj][0];
                    outai=outa[ii*ny+jj][1];
#endif
    

                    /* add contributions to the total shifted image at i,j 
                       noting that only the real part matters */
                    totalimij+= (outar*fftdeltr-outai*fftdelti);
                 }
             }
         
         /* record shifted image at i,j, after normalizing */
         outarr[i*ny+j] = (double) totalimij*normfac;
      }
    }

/* Free the plans and locally created arrays */

      fftw_free (outa);
      fftw_free (kx);
      fftw_free (ky);
      fftw_free (cskx);
      fftw_free (snkx);
      fftw_free (csky);
      fftw_free (snky);
      fftw_free (ina);
      fftw_destroy_plan (pa);
      if(verbose)
      {
         printf ("warp: finished\n");
         fflush(stdout);
      }

return 0;
}

void warp_frac2d_f77(double *arr, double *delx, double *dely, double *outarr,
     i4 *nx, i4*ny, i4 *transp, i4 *verbose)
{
  i4 ierwarp;
  ierwarp=warp_frac2d(arr, delx, dely, outarr, *nx, *ny, *transp, *verbose);
  return;
}

void warp_frac2d_f77_(double *arr, double *delx, double *dely, double *outarr,
     i4 *nx, i4*ny, i4 *transp, i4 *verbose)
{
  i4 ierwarp;
  ierwarp=warp_frac2d(arr, delx, dely, outarr, *nx, *ny, *transp, *verbose);
  return;
}

void warp_frac2d_f77__(double *arr, double *delx, double *dely, double *outarr,
     i4 *nx, i4*ny, i4 *transp, i4 *verbose)
{
  i4 ierwarp;
  ierwarp=warp_frac2d(arr, delx, dely, outarr, *nx, *ny, *transp, *verbose);
  return;
}

void shift_frac2d_f77(double *arr, double *delx, double *dely, double *outarr,
     i4 *nx, i4 *ny, i4 *transp, i4 *verbose)
{
  i4 iershift;
  iershift=shift_frac2d(arr, *delx, *dely, outarr, *nx, *ny, *transp, *verbose);
  return;
}

void shift_frac2d_f77_(double *arr, double *delx, double *dely, double *outarr,
     i4 *nx, i4 *ny, i4 *transp, i4 *verbose)
{
  i4 iershift;
  iershift=shift_frac2d(arr, *delx, *dely, outarr, *nx, *ny, *transp, *verbose);
  return;
}

void shift_frac2d_f77__(double *arr, double *delx, double *dely, double *outarr,
     i4 *nx, i4 *ny, i4 *transp, i4 *verbose)
{
  i4 iershift;
  iershift=shift_frac2d(arr, *delx, *dely, outarr, *nx, *ny, *transp, *verbose);
  return;
}