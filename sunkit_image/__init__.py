"""
sunkit-image
============

An open-source Python library for Solar Physics image processing.

Web Links
---------
Homepage: https://sunpy.org

Documentation: http://docs.sunpy.org/projects/sunkit-image/en/stable/
"""

__all__ = []


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
