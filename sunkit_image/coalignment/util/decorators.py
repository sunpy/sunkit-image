__all__ = ["register_coalignment_method", "registered_methods"]

# Global Dictionary to store the registered methods and their names
registered_methods = {}


def register_coalignment_method(name):
    """
    Registers a coalignment method to be used by the coalignment interface.

    Parameters
    ----------
    name : str
        The name of the coalignment method.
    """

    def decorator(func):
        registered_methods[name] = func
        return func

    return decorator
