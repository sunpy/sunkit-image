from sunkit_image.utils.decorators import register_coalignment_method, registered_methods


def test_register_coalignment_method():
    @register_coalignment_method("test_method")
    def test_func():
        return "Test function"

    assert "test_method" in registered_methods
    assert registered_methods["test_method"] == test_func
    assert test_func() == "Test function"
