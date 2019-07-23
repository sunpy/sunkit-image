import os
import platform
from glob import glob
from distutils.core import Extension

from astropy_helpers import setup_helpers


ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():

    if platform.system() == "Windows":
        return list()
    else:
        # 'numpy' will be replaced with the proper path to the numpy includes
        cfg = setup_helpers.DistutilsExtensionArgs()
        cfg["include_dirs"].append("numpy")
        cfg["include_dirs"].append("/usr/include/")
        cfg["sources"].extend(sorted(glob(os.path.join(ROOT, "src", "*.c"))))
        cfg["sources"].extend(sorted(glob(os.path.join(ROOT, "pyflct.pyx"))))
        cfg["libraries"].extend(["m", "fftw3"])
        cfg["extra_compile_args"].extend(["-std=c99", "-O3"])

        e = Extension("sunkit_image.flct._pyflct", **cfg)
        return [e]
