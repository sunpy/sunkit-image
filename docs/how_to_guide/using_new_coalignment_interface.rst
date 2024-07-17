.. _sunkit-image-how-to-guide-using-the-coalignment-interface:

*******************************
Using the Coalignment Interface
*******************************

This guide explains how to use the coalignment interface in the sunkit-image package to align solar images.
The :func:`~sunkit_image.coalignment.coalignment` function facilitates image coalignment using various registered methods.


The :func:`~sunkit_image.coalignment.coalignment` function performs image coalignment using a specified method that is registered using the decorator :func:`~sunkit_image.utils.register_coalignment_method`.
For registering a new method, please check :ref:`this <sunkit-image-how-to-guide-add-a-new-coalignment-method>` guide.

Refer to the docstring of :func:`~sunkit_image.coalignment.coalignment` for detailed information on the parameters, return values, and exceptions.

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

Below is an example of how to use the :func:`~sunkit_image.coalignment.coalignment` function:

.. code-block:: python

    from sunpy.map import Map
    from sunkit_image.coalignment.interface import coalignment
    import  sunpy.data.sample

    reference_map = Map(sunpy.data.sample.AIA_193_CUTOUT01_IMAGE)
    target_map = Map(sunpy.data.sample.AIA_193_CUTOUT02_IMAGE)

    coaligned_map = coalignment(reference_map, target_map, method="match_template")

The :func:`~sunkit_image.coalignment.coalignment` function aligns the ``target_map`` to the ``reference_map`` using the specified method (e.g., ``"match_template"``).

Registered Methods
==================

Ensure that the coalignment method you intend to use is registered.
You can add custom methods as described in :ref:`this <sunkit-image-how-to-guide-add-a-new-coalignment-method>` guide.

Handling NaNs and Infs
======================

The :func:`~sunkit_image.coalignment.coalignment` function includes a warning mechanism to alert users if there are any NaNs, Infs, or other problematic values in the input or template arrays.
Proper handling of these values is expected to be included in the registered methods.

