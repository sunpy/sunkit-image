# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import datetime
import os
import pathlib

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
from sunkit_image import __version__

release = __version__

project = "sunkit_image"
author = "The SunPy Community"
copyright = f"{datetime.datetime.now().year}, {author}"  # noqa: A001

# -- General configuration ---------------------------------------------------

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

# -- General configuration -----------------------------------------------------
suppress_warnings = [
    "app.add_directive",
]

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "sphinx_changelog",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
html_extra_path = ["robots.txt"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = ".rst"
master_doc = "index"

# Treat everything in single ` as a Python reference.
default_role = 'py:obj'


# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/", None)}

# -- Options for HTML output ---------------------------------------------------
graphviz_output_format = "svg"
graphviz_dot_args = [
    "-Nfontsize=10",
    "-Nfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Efontsize=10",
    "-Efontname=Helvetica Neue, Helvetica, Arial, sans-serif",
    "-Gfontsize=10",
    "-Gfontname=Helvetica Neue, Helvetica, Arial, sans-serif",
]

# -- Options for the Sphinx gallery -------------------------------------------
path = pathlib.Path.cwd()
example_dir = path.parent.joinpath("examples")
sphinx_gallery_conf = {
    "backreferences_dir": str(path.joinpath("generated", "modules")),
    "filename_pattern": "^((?!skip_).)*$",
    "examples_dirs": example_dir,
    "gallery_dirs": path.joinpath("generated", "gallery"),
    "default_thumb_file": path.joinpath("logo", "sunpy_icon_128x128.png"),
    "abort_on_example_error": False,
    "plot_gallery": "True",
    "remove_config_comments": True,
    "only_warn_on_example_error": True,
}
