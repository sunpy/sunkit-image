Coalignment
***********

``sunpy.coalignment`` provides routines to perform coalignment of solar images. The main entry point is the ``coalign`` function, which accepts a reference map, a target map, and a specified method
for coalignment. This method returns a new map with updated metadata reflecting the applied affine transformations, such as scaling, rotation, and translation. 
The module supports various transformation methods registered via the ``register_coalignment_method`` decorator, allowing for flexible coalignment strategies based on the specific needs of the data.

.. automodapi:: sunkit_image.coalignment.interface

.. automodapi:: sunkit_image.coalignment.match_template

.. automodapi:: sunkit_image.coalignment.decorators