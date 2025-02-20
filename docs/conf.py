"""
Configuration file for the Sphinx documentation builder.
"""

import datetime
import os
import pathlib
import warnings
from pathlib import Path

from packaging.version import Version

from astropy.utils.exceptions import AstropyDeprecationWarning
from matplotlib import MatplotlibDeprecationWarning
from sunpy.util.exceptions import SunpyDeprecationWarning, SunpyPendingDeprecationWarning
from sunpy_sphinx_theme import PNG_ICON

# -- Read the Docs Specific Configuration --------------------------------------

os.environ["PARFIVE_HIDE_PROGRESS"] = "True"
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    os.environ["SUNPY_CONFIGDIR"] = "/home/docs/"
    os.environ["HOME"] = "/home/docs/"
    os.environ["LANG"] = "C"
    os.environ["LC_ALL"] = "C"

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
from sunkit_image import __version__

_version = Version(__version__)
version = release = str(_version)
# Avoid "post" appearing in version string in rendered docs
if _version.is_postrelease:
    version = release = _version.base_version
# Avoid long githashes in rendered Sphinx docs
elif _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"
is_development = _version.is_devrelease
is_release = not(_version.is_prerelease or _version.is_devrelease)

project = "sunkit_image"
author = "The SunPy Community"
copyright = f"{datetime.datetime.now(datetime.UTC).year}, {author}"  # NOQA: A001

# -- General configuration -----------------------------------------------------

# Wrap large function/method signatures
maximum_signature_line_length = 80

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_changelog",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# Treat everything in single ` as a Python reference.
default_role = "py:obj"

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/python3.inv"),
    ),
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/numpy.inv"),
    ),
    "scipy": (
        "https://docs.scipy.org/doc/scipy/reference/",
        (None, "http://www.astropy.org/astropy-data/intersphinx/scipy.inv"),
    ),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sunpy": ("https://docs.sunpy.org/en/stable/", None),
    "ndcube": ('https://docs.sunpy.org/projects/ndcube/en/stable/', None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
}

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sunpy"

# Render inheritance diagrams in SVG
graphviz_output_format = "svg"

graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# By default, when rendering docstrings for classes, sphinx.ext.autodoc will
# make docs with the class-level docstring and the class-method docstrings,
# but not the __init__ docstring, which often contains the parameters to
# class constructors across the scientific Python ecosystem. The option below
# will append the __init__ docstring to the class-level docstring when rendering
# the docs. For more options, see:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"

# -- Other options ----------------------------------------------------------

html_extra_path = ["robots.txt"]

napoleon_use_rtype = False

napoleon_google_docstring = False

# Enable nitpicky mode, which forces links to be non-broken
nitpicky = True

# This is not used. See docs/nitpick-exceptions file for the actual listing.
nitpick_ignore = []
with Path("nitpick-exceptions").open() as f:
    for line in f.readlines():
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        target = target.strip()
        nitpick_ignore.append((dtype, target))

# We want to make sure all the following warnings fail the build
warnings.filterwarnings("error", category=SunpyDeprecationWarning)
warnings.filterwarnings("error", category=SunpyPendingDeprecationWarning)
warnings.filterwarnings("error", category=MatplotlibDeprecationWarning)
warnings.filterwarnings("error", category=AstropyDeprecationWarning)

# For the linkcheck
linkcheck_ignore = [
    r"https://doi.org/\d+",
    r"https://element.io/\d+",
    r"https://github.com/\d+",
    r"https://docs.sunpy.org/\d+",
]
linkcheck_anchors = False

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog = """
.. SunPy
.. _SunPy: https://sunpy.org
.. _`SunPy mailing list`: https://groups.google.com/group/sunpy
.. _`SunPy dev mailing list`: https://groups.google.com/group/sunpy-dev
"""

# -- Options for the Sphinx gallery -----------------------------------------

path = pathlib.Path.cwd()
example_dir = path.parent.joinpath("examples")
sphinx_gallery_conf = {
    "backreferences_dir": str(path.joinpath("generated", "modules")),
    "filename_pattern": "^((?!skip_).)*$",
    "examples_dirs": example_dir,
    "gallery_dirs": path.joinpath("generated", "gallery"),
    "default_thumb_file": PNG_ICON,
    "abort_on_example_error": False,
    "plot_gallery": "True",
    "remove_config_comments": True,
    "only_warn_on_example_error": True,
}
