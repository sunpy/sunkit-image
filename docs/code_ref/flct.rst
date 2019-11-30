****
FLCT
****

This subpackage contains routines which can be used to perform Fourier Local Correlation Tracking.
Using a C-based implementation of the FLCT algorithm developed by George H. Fisher and Brian T. Welsch, who we would like to thank for letting us it within sunkit-image.
The C code can be `found here <http://cgem.ssl.berkeley.edu/cgi-bin/cgem/FLCT/home>`__.
The following papers are references for the FLCT algorithm:

* `Welsch et al, ApJ 610, 1148, (2004) <https://iopscience.iop.org/article/10.1086/421767>`__
* `Fisher & Welsch, PASP 383, 373, (2008) <https://arxiv.org/abs/0712.4289>`__
* Fisher et al. ("The PDFI_SS Electric Field Inversion Software", in prep)

.. note::
    The FLCT is licensed under the GNU Lesser General Public License, version 2.1, see ``licenses/LICENSE_FLCT.rst``.

.. automodapi:: sunkit_image.flct
    :no-inheritance-diagram:
