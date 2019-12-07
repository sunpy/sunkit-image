# Licensed under GNU Lesser General Public License, version 2.1 - see licenses/LICENSE_FLCT.rst
import os
from glob import glob
from distutils.core import Extension

import numpy as np
from astropy_helpers import setup_helpers  # NOQA

ROOT = os.path.dirname(__file__)


def get_extensions():
    cfg = setup_helpers.DistutilsExtensionArgs()
    cfg["include_dirs"].append(np.get_include())
    cfg["include_dirs"].append("/usr/include/")
    cfg["sources"].extend(sorted(glob(os.path.join(ROOT, "src", "*.c"))))
    cfg["sources"].extend(sorted(glob(os.path.join(ROOT, "src", "pyflct.pyx"))))
    cfg["libraries"].extend(["m", "fftw3"])

    if setup_helpers.get_compiler_option() == "msvc":
        return list()
    else:
        cfg["extra_compile_args"].extend(["-O3", "-Wall", "-fomit-frame-pointer", "-fPIC"])

    return [Extension("sunkit_image.flct._pyflct", **cfg)]
