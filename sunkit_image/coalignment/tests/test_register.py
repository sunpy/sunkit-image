from sunkit_image.coalignment import coalign
from sunkit_image.coalignment.register import REGISTERED_METHODS, register_coalignment_method


def test_register_coalignment_method():
    @register_coalignment_method("test_method")
    def test_func():
        return "Test function"

    assert "test_method" in REGISTERED_METHODS
    assert REGISTERED_METHODS["test_method"] == test_func
    assert test_func() == "Test function"


def test_coalignment_method_in_docstring():
    assert (
        "phase_cross_correlation" in coalign.__doc__
    )
