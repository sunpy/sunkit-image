import inspect
from typing import Union, Callable

import numpy as np

from sunpy.map import GenericMap, Map


def accept_array_or_map(*, arg_name: str) -> Callable[Callable, Callable]:
    """
    Decorator that allows a function to accept an array or a
    `sunpy.map.GenericMap` as an argument.

    This can be applied to functions that:
    - Take a single array or map as input
    - Return a single array that has the same pixel coordinates
      as the input array.

    If the decorated function is given a map, this decorator will
    automatically handle returning a map, with new data and the
    original metadata.
    """

    def decorate(f: Callable) -> Callable:
        sig = inspect.signature(f)
        if arg_name not in sig.parameters:
            raise RuntimeError(f"Could not find '{arg_name}' in function signature")

        def inner(*args, **kwargs) -> Union[np.ndarray, GenericMap]:
            sig_bound = sig.bind(*args, **kwargs)
            map_arg = sig_bound.arguments[arg_name]
            if isinstance(map_arg, GenericMap):
                map_in = True
                sig_bound.arguments[arg_name] = map_arg.data
            elif isinstance(map_arg, np.ndarray):
                map_in = False
            else:
                raise ValueError(f"'{arg_name}' argument must be a sunpy map or numpy array (got type {type(map_arg)})")

            # Run decorated function
            array_out = f(*sig_bound.args, **sig_bound.kwargs)

            if map_in:
                return Map(array_out, map_arg.meta)
            else:
                return array_out

        return inner

    return decorate
