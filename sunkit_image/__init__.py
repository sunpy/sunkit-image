"""
sunkit-image
============

An open-source Python library for Solar Physics image processing.

Web Links
---------
Homepage: https://sunpy.org
Documentation: https://docs.sunpy.org/en/stable/
"""
# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import os
import sys

__minimum_python_version__ = "3.6"


class UnsupportedPythonError(Exception):
    pass


if sys.version_info < tuple(
    (int(val) for val in __minimum_python_version__.split("."))
):
    raise UnsupportedPythonError(
        f"sunpy does not support Python < {__minimum_python_version__}"
    )

# this indicates whether or not we are in the package's setup.py
try:
    _SUNPY_SETUP_
except NameError:
    import builtins

    builtins._SUNPY_SETUP_ = False

try:
    from .version import version as __version__
except ImportError:
    __version__ = ""
try:
    from .version import githash as __githash__
except ImportError:
    __githash__ = ""
