__all__ = ["__version__", "__githash__"]

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


if not _SUNPY_SETUP_:
    import os
    from sunpy.tests.runner import SunPyTestRunner

    self_test = SunPyTestRunner.make_test_runner_in(os.path.dirname(__file__))
    __all__ += ["self_test"]
