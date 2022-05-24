from pathlib import Path
from functools import wraps

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

import sunpy
from sunpy.tests.helpers import skip_windows  # NOQA

import sunkit_image


def get_hash_library_name():
    """
    Generate the hash library name for this env.
    """
    ft2_version = f"{mpl.ft2font.__freetype_version__.replace('.', '')}"
    mpl_version = "dev" if "+" in mpl.__version__ else mpl.__version__.replace(".", "")
    sunkit_image_version = "dev" if "dev" in sunkit_image.__version__ else sunpy.__version__.replace(".", "")
    sunpy_version = "dev" if "dev" in sunpy.__version__ else sunpy.__version__.replace(".", "")
    return f"figure_hashes_mpl_{mpl_version}_ft_{ft2_version}_sunkit_image_{sunkit_image_version}_sunpy_{sunpy_version}.json"


def figure_test(test_function):
    """
    A decorator which marks the test as comparing the hash of the returned
    figure to the hash library in the repository. A `matplotlib.figure.Figure`
    object should be returned or ``plt.gcf()`` will be called to get the figure
    object to compare to.

    Examples
    --------
    .. code::
        @figure_test
        def test_simple_plot():
            plt.plot([0,1])
    """
    hash_library_name = get_hash_library_name()
    hash_library_file = Path(__file__).parent / hash_library_name

    @pytest.mark.remote_data
    @pytest.mark.mpl_image_compare(
        hash_library=hash_library_file, savefig_kwargs={"metadata": {"Software": None}}, style="default"
    )
    @wraps(test_function)
    def test_wrapper(*args, **kwargs):
        ret = test_function(*args, **kwargs)
        if ret is None:
            ret = plt.gcf()
        return ret

    return test_wrapper
