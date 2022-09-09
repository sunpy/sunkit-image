"""
=========================
Creating a Carrington Map
=========================
This example shows how to reproject a single image of the Sun to a heliographic
Carrington coordinate system.

This uses the `sunkit_image.reproject.carrington_header` function to create a
new map header that represents a Carrington coordinate frame, and then
reprojects the AIA map to that new coordinate frame.
"""
import matplotlib.pyplot as plt

import sunpy.data.sample
import sunpy.map

from sunkit_image.reproject import carrington_header

aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

new_header = carrington_header(aia_map, shape_out=[360, 720])
print(new_header)

carrington_map = aia_map.reproject_to(new_header)


fig = plt.figure()

ax = plt.subplot(2, 1, 1, projection=aia_map)
aia_map.plot(vmin=1, vmax=5e4)
ax.set_title("Helioprojective")

ax = plt.subplot(2, 1, 2, projection=carrington_map)
carrington_map.plot(vmin=1, vmax=5e4)
ax.set_title("Heliographic Carrington")

fig.subplots_adjust(hspace=0.5)

plt.show()
