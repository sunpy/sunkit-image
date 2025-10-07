__all__ = [
    "REGISTERED_METHODS",
    "register_coalignment_method",
]

# Global Dictionary to store the registered methods and their names
REGISTERED_METHODS = {}

def register_coalignment_method(name):
    """
    Registers a coalignment method to be used by the coalignment interface.

    Parameters
    ----------
    name : str
        The name of the coalignment method.
    """

    def decorator(func):
        REGISTERED_METHODS[name] = func
        return func

    return decorator
