.. _sunkit-image-how-to-guide-using-the-coalignment-interface:

*******************************
Using the Coalignment Interface
*******************************

This guide explains how to use the coalignment interface to improve the alignment of solar images.
The :func:`~sunkit_image.coalignment.coalignment` function facilitates image coalignment using a range of  registered methods.
Refer to the docstring of :func:`~sunkit_image.coalignment.coalignment` for detailed information on the parameters, return values, and exceptions.

Here is an example of how to use the :func:`~sunkit_image.coalignment.coalignment` function:

.. code-block:: python

    from sunpy.map import Map
    from sunkit_image.coalignment.interface import coalignment
    import  sunpy.data.sample

    reference_map = Map(sunpy.data.sample.AIA_193_CUTOUT01_IMAGE)
    target_map = Map(sunpy.data.sample.AIA_193_CUTOUT02_IMAGE)

    coaligned_map = coalignment(reference_map, target_map, method="match_template")

There is another example (link to the EIS and AIA gallery example).

