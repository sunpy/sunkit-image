import warnings

import astropy.units as u
import numpy as np
import sunpy.map
from skimage.feature import match_template
from sunpy.util.exceptions import SunpyUserWarning


## This dictionary will be further replaced in a more appropriate location, once the decorator structure is in place.
registered_methods = {}


def register_coalignment_method(name, method):
    """
    Registers a coalignment method to be used by the coalignment interface.

    Parameters
    ----------
    name : str
        The name of the coalignment method.
    method : callable
        The function implementing the coalignment method.
    """
    registered_methods[name] = method
