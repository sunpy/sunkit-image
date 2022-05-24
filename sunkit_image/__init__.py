"""
sunkit-image
============

A image processing toolbox for Solar Physics.

* Homepage: https://sunpy.org
* Documentation: https://sunkit-image.readthedocs.io/en/latest/
"""
import sys

from .version import version as __version__

# Enforce Python version check during package import.
__minimum_python_version__ = "3.7"


class UnsupportedPythonError(Exception):
    """
    Running on an unsupported version of Python.
    """


if sys.version_info < tuple(int(val) for val in __minimum_python_version__.split(".")):
    # This has to be .format to keep backwards compatibly.
    raise UnsupportedPythonError("sunkit_image does not support Python < {}".format(__minimum_python_version__))

__all__ = []
