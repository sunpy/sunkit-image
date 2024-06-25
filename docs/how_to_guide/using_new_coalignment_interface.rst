.. _sunkit-image-how-to-guide-using-the-coalignment-interface:

*****************************
Using the Coalignment Interface
*****************************

This guide explains how to use the coalignment interface in the `sunkit_image` package to align solar images. The :func:`sunkit_image.coalignment_module.interface.coalignment` function facilitates image coalignment using various registered methods.

Function Overview
=================
The :func:`sunkit_image.coalignment_module.interface.coalignment` function performs image coalignment using a specified method that is registered using the decorator :func:`sunkit_image.coalignment_module.util.decorators.register_coalignment_method`. For registering a new method, please check the :ref:`_sunkit-image-how-to-guide-adding-a-new-coalignment-method` guide.

Refer to the docstring of :func:`sunkit_image.coalignment_module.interface.coalignment` for detailed information on the parameters, return values, and exceptions.

Function Parameters
===================
- **reference_map** (`sunpy.map.Map`): The reference map to which the target map is to be coaligned.
- **target_map** (`sunpy.map.Map`): The target map to be coaligned to the reference map.
- **method** (str): The name of the registered coalignment method to use.

Returns
=======
- **sunpy.map.Map**: The coaligned target map.

Raises
======
- **ValueError**: If the specified method is not registered.

Example Usage
=============
Below is an example of how to use the :func:`sunkit_image.coalignment_module.interface.coalignment` function:

.. code-block:: python

    from sunpy.map import Map
    from sunkit_image.coalignment_module.interface import coalignment
    import  sunpy.data.sample

    reference_map = Map(sunpy.data.sample.AIA_193_CUTOUT01_IMAGE)
    target_map = Map(sunpy.data.sample.AIA_193_CUTOUT02_IMAGE)

    coaligned_map = coalignment(reference_map, target_map, method="match_template")

The :func:`sunkit_image.coalignment_module.interface.coalignment` function aligns the ``target_map`` to the ``reference_map`` using the specified method (e.g., ``"match_template"``).

Registered Methods
==================
Ensure that the coalignment method you intend to use is registered. You can add custom methods as described in :ref:`_sunkit-image-how-to-guide-adding-a-new-coalignment-method`.

Handling NaNs and Infs
======================
The :func:`sunkit_image.coalignment_module.interface.coalignment` function includes a warning mechanism to alert users if there are any NaNs, Infs, or other problematic values in the input or template arrays. Proper handling of these values is expected to be included in the registered methods.

Further Reading
===============
For more details on how to register new coalignment methods, refer to :ref:`_sunkit-image-how-to-guide-adding-a-new-coalignment-method`.
