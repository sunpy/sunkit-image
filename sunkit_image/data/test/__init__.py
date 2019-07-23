import os
import re
import glob
import fnmatch

from astropy.utils.data import get_pkg_data_filename

import sunkit_image

__all__ = ["rootdir", "file_list", "get_test_filepath", "test_data_filenames"]

rootdir = os.path.join(os.path.dirname(sunkit_image.__file__), "data")
file_list = glob.glob(os.path.join(rootdir, "*.[!p]*"))


def get_test_filepath(filename, **kwargs):
    """
    Return the full path to a test file in the ``data`` directory.

    Parameters
    ----------
    filename : `str`
        The name of the file inside the ``data`` directory.

    Return
    ------
    filepath : `str`
        The full path to the file.

    Notes
    -----

    This is a wrapper around `~astropy.utils.data.get_pkg_data_filename` which
    sets the ``package`` kwarg to be ``sunkit_image.data.test``.
    """
    return get_pkg_data_filename(filename, package="sunkit_image.data.test", **kwargs)


def test_data_filenames():
    """
    Return a list of all test files in ``data`` directory.

    This ignores any ``py``, ``pyc`` and ``__*__`` files in these directories.

    Return
    ------
    get_all_test_filepath : `list`
        The name of all test files in ``data/test`` directory.
    """
    test_data_filenames_list = []
    excludes = ["*.pyc", "*" + os.path.sep + "__*__", "*.py"]
    excludes = r"|".join([fnmatch.translate(x) for x in excludes]) or r"$."

    for root, dirs, files in os.walk(rootdir):
        files = [os.path.join(root, f) for f in files]
        files = [f for f in files if not re.match(excludes, f)]
        files = [file.replace(rootdir + os.path.sep, "") for file in files]
        test_data_filenames_list.extend(files)

    return test_data_filenames_list
