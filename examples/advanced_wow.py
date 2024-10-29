"""
====================================================
Advanced Usage of Wavelets Optimized Whitening (WOW)
====================================================

This example demonstrates different options of the Wavelets Optimized Whitening applied to a `sunpy.map.Map`
using `sunkit_image.enhance.wow`.
"""

import matplotlib.pyplot as plt

import sunpy.data.sample
import sunpy.map

import sunkit_image.enhance as enhance

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, LinearStretch, PowerStretch, AsymmetricPercentileInterval

###########################################################################
# `sunpy` provides a range of sample data with a number of suitable images.
# Here will just use AIA 193.

input_map = sunpy.map.Map(sunpy.data.sample.AIA_193_JUN2012)

###########################################################################
# We will now crop the southwestern quadrant.

top_right = SkyCoord(1200 * u.arcsec, 0 * u.arcsec, frame=input_map.coordinate_frame)
bottom_left = SkyCoord(0 * u.arcsec, -1200 * u.arcsec, frame=input_map.coordinate_frame)
input_map = input_map.submap(bottom_left, top_right=top_right)

###########################################################################
# We now will apply different options of the Wavelets Optimized Whitening algorithm.
# The `sunkit_image.enhance.wow` function takes a `sunpy.map.Map` as an input.
# First, we call WOW with no arguments, which returns the default WOW enhancement.

wow_map = enhance.wow(input_map)

###########################################################################
# Then we can denoise the output using a soft threshold in the three first wavelet
# scales using "sigma = 5, 2, 1".

denoise_coefficients = [5, 2, 1]
wow_map_denoised = enhance.wow(input_map,denoise_coefficients=denoise_coefficients)

###########################################################################
# We then run the edge-aware (bilateral) flavor of the algorithm.
# This prevents ringing around sharp edges (e.g., the solar limb 
# or very bright features.

wow_map_bilateral = enhance.wow(input_map, bilateral=1)

###########################################################################
# This will call the edge-aware algorithm with denoising.

wow_map_bilateral_denoised = enhance.wow(input_map, bilateral=1, denoise_coefficients=denoise_coefficients)

###########################################################################
# Finally, we merge the denoised edge-aware enhanced image with the
# gamma-stretched input, with weight "h".

gamma = 4
wow_map_bilateral_denoised_merged = enhance.wow(input_map, bilateral=1, denoise_coefficients=denoise_coefficients, gamma=gamma, h=0.99)

###########################################################################
# Finally, we will plot the full set of outputs created and
# compare that to the original image.

fig = plt.figure(figsize=(8, 12))
variations = {
    'Input | gamma = {gamma} stretch': {'map': input_map, 'stretch': PowerStretch(1 / gamma)},
    'WOW | linear stretch': {'map': wow_map, 'stretch': LinearStretch()},
    'denoised WOW': {'map': wow_map_denoised, 'stretch': LinearStretch()},
    'Edge-aware WOW': {'map': wow_map_bilateral, 'stretch': LinearStretch()},
    'Edge-aware & denoised WOW': {'map': wow_map_bilateral_denoised, 'stretch': LinearStretch()},
    'Merged with input': {'map': wow_map_bilateral_denoised_merged, 'stretch': LinearStretch()}
}
interval = AsymmetricPercentileInterval(1, 99.9)
for i, (title, image) in enumerate(variations.items()):
    ax = fig.add_subplot(3, 2, i + 1, projection=image['map'])
    image['map'].plot(norm=ImageNormalize(image['map'].data, interval=interval, stretch=image['stretch']))
    ax.set_title(title)
    ax.axis('off')

fig.tight_layout()

plt.show()
