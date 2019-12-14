# Licensed under GNU Lesser General Public License, version 2.1 - see licenses/LICENSE_FLCT.rst
import os
import sys
from glob import glob
from collections import defaultdict
from distutils.core import Extension

import numpy as np
from extension_helpers import get_compiler

ROOT = os.path.dirname(__file__)


def get_extensions():
    cfg = defaultdict(list)
    cfg["include_dirs"].append(np.get_include())
    cfg["include_dirs"].append("/usr/include/")
    cfg["include_dirs"].append(os.path.join(sys.prefix, "include"))
    cfg["sources"].extend(sorted(glob(os.path.join(ROOT, "src", "*.c"))))
    cfg["sources"].extend(sorted(glob(os.path.join(ROOT, "src", "pyflct.pyx"))))
    cfg["libraries"].extend(["m", "fftw3"])

    if get_compiler() == "msvc":
        return list()
    else:
        cfg["extra_compile_args"].extend(["-O3", "-Wall", "-fomit-frame-pointer", "-fPIC"])

    return [Extension("sunkit_image.flct._pyflct", **cfg)]
