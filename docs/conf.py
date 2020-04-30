# flake8: NOQA
import os
import sys
import datetime
from configparser import ConfigParser

try:
    import sphinx_gallery

    sphinx_gallery.__version__
    has_sphinx_gallery = True
except ImportError:
    has_sphinx_gallery = False


conf = ConfigParser()
conf.read([os.path.join(os.path.dirname(__file__), "..", "setup.cfg")])
setup_cfg = dict(conf.items("metadata"))

# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = setup_cfg["name"]
author = setup_cfg["author"]
copyright = "{0}, {1}".format(datetime.datetime.now().year, setup_cfg["author"])

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
__import__(setup_cfg["name"])
package = sys.modules[setup_cfg["name"]]

# The short X.Y version.
version = ".".join(package.__version__.split(".")[:3])
# The full version, including alpha/beta/rc tags.
release = package.__version__.split("+")[0]
# Is this version a development release
is_development = ".dev" in release

try:
    from sunpy_sphinx_theme.conf import *

    html_sidebars = {"**": ["docsidebar.html"]}
except ImportError:
    html_theme = "default"

html_title = "{0} v{1}".format(project, release)
htmlhelp_basename = project + "doc"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
]

# -- Options for the Sphinx gallery -------------------------------------------
if has_sphinx_gallery:
    import pathlib

    extensions += ["sphinx_gallery.gen_gallery"]
    path = pathlib.Path.cwd()
    example_dir = path.parent.joinpath("examples")
    sphinx_gallery_conf = {
        "backreferences_dir": str(path.joinpath("generated", "modules")),
        "filename_pattern": "^((?!skip_).)*$",
        "examples_dirs": example_dir,
        "gallery_dirs": path.joinpath("generated", "gallery"),
        "default_thumb_file": path.joinpath("logo", "sunpy_icon_128x128.png"),
        "abort_on_example_error": False,
        "plot_gallery": True,
    }

"""
Write the latest changelog into the documentation.
"""
target_file = os.path.abspath("./whatsnew/latest_changelog.txt")
try:
    from sunpy.util.towncrier import generate_changelog_for_docs

    if is_development:
        generate_changelog_for_docs("../", target_file)
except Exception as e:
    print(f"Failed to add changelog to docs with error {e}.")
# Make sure the file exists or else sphinx will complain.
open(target_file, "a").close()


def setup(app):
    if not has_sphinx_gallery:
        import warnings

        warnings.warn(
            "The sphinx_gallery extension is not installed, so the "
            "gallery will not be built. You will probably see "
            "additional warnings about undefined references due "
            "to this."
        )
