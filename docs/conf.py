import os
import sys
import datetime
# Get configuration information from setup.cfg
from configparser import ConfigParser

try:
    from sphinx_astropy.conf.v1 import *  # noqa
except ImportError:
    print(
        "ERROR: the documentation requires the sphinx-astropy package to be installed"
    )
    sys.exit(1)

try:
    import sphinx_gallery

    if on_rtd and os.environ.get("READTHEDOCS_PROJECT").lower() != "sunpy":
        # Gallery takes too long on RTD to build unless you have extra build time.
        has_sphinx_gallery = False
    else:
        has_sphinx_gallery = True
except ImportError:
    has_sphinx_gallery = False

if on_rtd:
    os.environ["SUNPY_CONFIGDIR"] = "/home/docs/"
    os.environ["HOME"] = "/home/docs/"
    os.environ["LANG"] = "C"
    os.environ["LC_ALL"] = "C"


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

# -- Options for the edit_on_github extension ---------------------------------
if eval(setup_cfg.get("edit_on_github")):
    extensions += ["sphinx_astropy.ext.edit_on_github"]

    edit_on_github_project = setup_cfg["github_project"]
    if "v" in release:
        edit_on_github_branch = "v" + release
    else:
        edit_on_github_branch = "master"

    edit_on_github_source_root = ""
    edit_on_github_doc_root = "docs"

# -- Resolving issue number to links in changelog -----------------------------
github_issues_url = "https://github.com/{0}/issues/".format(setup_cfg["github_project"])

# -- Options for the Sphinx gallery -------------------------------------------
if has_sphinx_gallery:
    import pathlib

    extensions += ["sphinx_gallery.gen_gallery"]
    path = pathlib.Path.cwd()
    example_dir = path.parent.joinpath("examples")
    sphinx_gallery_conf = {
        "backreferences_dir": path.joinpath("generated", "modules"),
        "filename_pattern": "^((?!skip_).)*$",
        "examples_dirs": example_dir,
        "gallery_dirs": path.joinpath("generated", "gallery"),
        "default_thumb_file": path.joinpath("logo", "sunpy_icon_128x128.png"),
        "abort_on_example_error": True,
        "plot_gallery": True,
    }

# Write the latest changelog into the documentation.
target_file = os.path.abspath("./whatsnew/latest_changelog.txt")
try:
    from sunpy.util.towncrier import generate_changelog_for_docs

    generate_changelog_for_docs("../", target_file)
    if is_development:
        generate_changelog_for_docs("../", target_file)
except Exception:
    # If we can't generate it, we need to make sure it exists or else sphinx
    # will complain.
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
