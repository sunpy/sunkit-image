from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize

libs = ["m", "fftw3"]
args = ["-std=c99", "-O3"]
sources = ["./src/flctsubs.c", "pyflctsubs.pyx"]
include = ["/usr/include/", numpy.get_include()]
linkerargs = ["-Wl,-rpath,lib"]
libdirs = ["lib"]


extensions = [
    Extension(
        "pyflctsubs",
        sources=sources,
        include_dirs=include,
        libraries=libs,
        library_dirs=libdirs,
        extra_compile_args=args,
        extra_link_args=linkerargs,
    )
]

setup(name="pyflctsubs", packages=["pyflctsubs"], ext_modules=cythonize(extensions))
