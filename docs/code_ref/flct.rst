****
FLCT
****

This subpackage contains routines which can be used to perform Fourier Local Correlation Tracking.
The FLCT is licensed under the GNU Lesser General Public License, version 2.1, see ``licenses/LICENSE_FLCT.rst``.

We would like to thank George H. Fisher and Brian T. Welsch for letting us include their C code within sunkit-image.
The following papers are references for the FLCT algorithm:

* Welsch et al, ApJ 610, 1148, (2004)
* Fisher & Welsch, PASP 383, 373, (2008)
* Fisher et al. ("The PDFI_SS Electric Field Inversion Software", in prep)

This subpackage wraps a C-based implementation of the FLCT algorithm. 
The C code is provided by G. H. Fisher and B. Welsch `here <http://solarmuri.ssl.berkeley.edu/~fisher/public/software/FLCT/C_VERSIONS/>`__. 
For further information see Welsch et al, 2004, ApJ, 610 (doi:10.1086/421767) and Fisher & Welsch 2008, 383, 373, Astronomical Society of the Pacific Conference Series (`<https://arxiv.org/abs/0712.4289>`__).

.. automodapi:: sunkit_image.flct
    :no-inheritance-diagram:

