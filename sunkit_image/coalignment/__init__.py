"""
Routines to perform coalignment of solar images
"""
# Need to register the the coalignment functions
from sunkit_image.coalignment.match_template import match_template_coalign  # isort:skip
from sunkit_image.coalignment.phase_cross_correlation import phase_cross_correlation_coalign  # isort:skip

from sunkit_image.coalignment.interface import coalign

__all__ = ["coalign"]
