.. _sunkit-image-how-to-guide-using-the-coalignment-interface:

*******************************
Using the Coalignment Interface
*******************************

This guide explains how to use the coalignment interface to improve the alignment of solar images.
The :func:`~sunkit_image.coalignment.coalign` function facilitates image coalignment using a range of registered methods.
Refer to the docstring of :func:`~sunkit_image.coalignment.coalign` for detailed information on the parameters, return values, and exceptions.

Here is an example of how to use the :func:`~sunkit_image.coalignment.coalign` function:

.. plot::
    :include-source:
    :context: close-figs

    import numpy as np
    from sunpy.map import Map
    from sunkit_image.coalignment import coalign
    import sunpy.data.sample
    import matplotlib.pyplot as plt
    # Load the AIA images
    reference_map = Map(sunpy.data.sample.AIA_193_IMAGE)
    target_map = Map(sunpy.data.sample.AIA_193_CUTOUT03_IMAGE)
    ## Should match the platescale before coalignment so as to maintain correctness;
    nx= (reference_map.scale.axis1 * reference_map.dimensions.x )/target_map.scale.axis1
    ny= (reference_map.scale.axis2 * reference_map.dimensions.y )/target_map.scale.axis2
    aia_193_downsampled_map = reference_map.resample(u.Quantity([nx,ny]))
    # Coalign the target map to the reference map
    coaligned_map = coalign(aia_193_downsampled_map, target_map, method="match_template")
    # Define contour levels
    levels = np.linspace(200, 1200, 5) * target_map.unit
    # Plotting
    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(121, projection=target_map)
    target_map.plot(axes=ax, title='Before coalignment')
    bounds = ax.axis()
    aia_193_downsampled_map.draw_contours(levels, axes=ax, cmap='viridis', alpha=0.7)
    ax.axis(bounds)
    ax = fig.add_subplot(122, projection=coaligned_map)
    coaligned_map.plot(axes=ax, title='After coalignment')
    bounds = ax.axis()
    aia_193_downsampled_map.draw_contours(levels, axes=ax, cmap='viridis', alpha=0.7)
    ax.axis(bounds)

    plt.show()

There is another example :ref:`sphx_glr_generated_gallery_aligning_aia_with_eis_maps.py` focused on aligning an EIS raster with an AIA image.
