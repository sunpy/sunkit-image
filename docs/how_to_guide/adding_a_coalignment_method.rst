.. _sunkit-image-how-to-guide-add-a-new-coalignment-method:

****************************
Add a New Coalignment Method
****************************

In addition to the coalignment methods provided in sunkit-image, users can also write their own
coalignment methods and "register" these methods with sunkit-image such that they can be used through the same interface as the builtin methods without having to alter the underlying
sunkit-image package.

At a minimum, your new coalignment function should do the following:

1. Take the following inputs:

   - ``target_array`` : The 2D array to be coaligned.
   - ``reference_array`` : The 2D array to align to.
   - ``**kwargs``: Optional keyword arguments used by your method

2. Decide the values of the affine transformation - translation, scale and rotation. In most cases, this means calculating the shifts in the x- and y-directions needed to align ``input_array`` with ``target_array``.

3. Return an instance of `~sunkit_image.coalignment.interface.AffineParams` with the results of your coalignemtn procedure.

Additionally, registered methods are expected to handled NaNs and Infs should they arise as a result of your coalignment procedure.
The :func:`~sunkit_image.coalignment.coalign` function does not make any attempt to filter out
these non-finite values.

To register your new coalignment method, you can use the :func:`~sunkit_image.coalignment.register.register_coalignment_method` decorator to register your new method with a custom name. An example of how to use this decorator is shown below:

.. code-block:: python

    from sunkit_image.coalignment.interface import AffineParams, register_coalignment_method

    @register_coalignment_method("my_custom_coalignment_method")
    def my_coalignment_method(target_array, reference_array, **kwargs):
        # Your coalignment code goes here
        # This should encompass calculating the shifts,
        # handling NaN values appropriately.
        # Return the shifts in an affine style, such as the scale, rotation and translation.
        return AffineParams(scale, rotation, translation)


To check if your method is registered, you can check if it is present in the registered methods dictionary ``REGISTERED_METHODS`` using the following code:

.. code-block:: python

    from sunkit_image.coalignment.interface import REGISTERED_METHODS

    print(REGISTERED_METHODS)

If your coalignment method has been successfully registered, you should now be able to call it
through the `~sunkit_image.coalignment.coalign` interface:

.. code-block:: python

        coaligned_map = coalign(target_map, reference_map, method='my_custom_coalignment_method')
