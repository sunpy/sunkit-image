.. _sunkit-image-how-to-guide-add-a-new-coalignment-method:

****************************
Add a New Coalignment Method
****************************

You can add a custom coalignment method in the sunkit-image package using the decorator :func:`~sunkit_image.utils.decorators.register_coalignment_method`:

.. code-block:: python

    from sunkit_image.utils.decorators import register_coalignment_method

    @register_coalignment_method("my_coalign")
    def my_coalignment_method(input_array, template_array):
        # Your coalignment code goes here
        # This should encompass calculating the shifts,
        # handling NaN values appropriately.
        # Return the shifts in a affine style, such as the scale, rotation and translation.
        return affine_params(scale, rotation, translation)

Decorator Parameters
====================

- **"my_coalign"**: The name of your custom coalignment method.

Function Requirements
=====================

Your coalignment function should:

1. **Take Input Parameters**:

   - ``input_array``: The 2D array to be coaligned.
   - ``template_array``: The 2D template array to align to.

2. **Compute Shifts**: Calculate the shifts in the x and y directions needed to align ``input_array`` with ``template_array``.

3. **Determine Affine Parameters**: Decide the parameters of the affine parameters like the scale, rotation and translation(generally shifts in x and y direction).

4. **Return**: A named tuple that contains the affine transformation parameters.

Example Usage
=============

Once you have added your custom coalignment method, you can use it as discussed in :ref:`sunkit-image-how-to-guide-using-the-coalignment-interface`.
