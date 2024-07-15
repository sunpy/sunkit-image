from sunkit_image.coalignment.interface import coalignment

# To extend the package with new coalignment methods, please follow the import style shown below.
from sunkit_image.coalignment.match_template import match_template_coalign  # noqa: F401

__all__ = ["coalignment"]
