from sunkit_image.coalignment.decorators import register_coalignment_method, registered_methods
from sunkit_image.coalignment.interface import affine_params, coalign
from sunkit_image.coalignment.match_template import match_template_coalign as _

__all__ = ["coalignment", "register_coalignment_method", "registered_methods", "affine_params"]
