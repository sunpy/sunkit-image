.. _sunkit-image-how-to-guide-add-a-new-coalignment-method:

****************************
Add a New Coalignment Method
****************************

If you want to register a new coalignment method that can be used by :func:`~sunkit_image.coalignment.coalign`, you can use :func:`~sunkit_image.coalignment.register.register_coalignment_method`:

.. code-block:: python

    from sunkit_image.coalignment.interface import AffineParams, register_coalignment_method

    @register_coalignment_method("my_coalign")
    def my_coalignment_method(input_array, target_array, **kwargs):
        # Your coalignment code goes here
        # This should encompass calculating the shifts,
        # handling NaN values appropriately.
        # Return the shifts in an affine style, such as the scale, rotation and translation.
        return AffineParams(scale, rotation, translation)

Decorator Parameters
====================

Currently the decorator takes one parameter:

- ``name`` : The name of your custom coalignment method, which in the above example is  "my_coalign".

Function Requirements
=====================

Your coalignment function should:

1. **Take Input Parameters**:

   - ``input_array`` : The 2D array to be coaligned.
   - ``target_array`` : The 2D array to align to.
   - ``**kwargs``: So extra arguments can be passed down the stack.

2. **Compute Shifts** : Calculate the shifts in the x and y directions needed to align ``input_array`` with ``target_array``.

3. **Determine Affine Parameters** : Decide the values of the affine parameters - translation, scale and rotation.

4. **Return** : Use the ``AffineParams`` named tuple included or provide your own that exposes the three parameters as attributes.

Handling NaNs and Infs
======================

Proper handling of these values is expected to be included in the registered methods.
The :func:`~sunkit_image.coalignment.coalign` function does not change any problematic values.

Checking if the Method is Registered
====================================

To check if your method is registered, you can check if it is present in the registered methods dictionary ``REGISTERED_METHODS`` using the following code:

.. code-block:: python

    from sunkit_image.coalignment.interface import REGISTERED_METHODS

    print(REGISTERED_METHODS)
