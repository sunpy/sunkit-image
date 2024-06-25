.. _sunkit-image-how-to-guide-add-a-new-coalignment-method:

****************************
Add a New Coalignment Method
****************************

You can add a custom coalignment method in the sunkit-image package using the decorator :func:`sunkit_image.coalignment_module.register_coalignment_method`:

.. code-block:: python

    from sunkit_image.coalignment_module.util.decorators import register_coalignment_method

    @register_coalignment_method("my_coalign")
    def my_coalignment_method(input_array, template_array):
        # Your coalignment code goes here
        # This should encompass calculating the shifts, applying these shifts to the data,
        # handling NaN values appropriately, and implementing any necessary clipping logic.
        # Return the shifts and the coaligned array
        return (coaligned_array, (x_shift, y_shift))

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

3. **Apply Shifts**: Apply these shifts to ``input_array`` to generate the coaligned array.

4. **Return**: A tuple where the first element is the coaligned array and the second element is another tuple containing the shifts ``(x_shift, y_shift)``.

Example Usage
=============

Once you have added your custom coalignment method, you can use it as discussed in :ref:`this <sunkit-image-how-to-guide-using-the-coalignment-interface>` guide
