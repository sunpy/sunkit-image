import matplotlib.pyplot as plt
import sunpy.data.sample
from sunpy.map import Map

from sunkit_image.enhance import wow

###############################################################################
# We will use the AIA 171 image from the sample data at the following

maps = Map(sunpy.data.sample.AIA_171_IMAGE)

# Then we create a map from a WOW-ed version of the image. We use the bilateral flavor of the algorithm, and
# denoising coefficients in the first three wavelet planes equal to 5, 2, & 1 sigma of the local noise
# The noise is estimated automatically. It is possible to pass a noise map for optimal results.

wow_map = Map(wow(map.data, bilateral=1, denoise_coefficients=[5, 2, 1]), map.meta)

fig = plt.figure()
ax = fig.add_subplot(111, projection=map.wcs)
im = ax.imshow(wow_map)
lon, lat = ax.coords
lon.set_axislabel("Helioprojective Longitude")
lat.set_axislabel("Helioprojective Latitude")
ax.set_title("WOW-ed AIA 171 image")
plt.show()
