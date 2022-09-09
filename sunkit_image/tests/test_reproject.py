import matplotlib.pyplot as plt

import sunpy.data.sample
import sunpy.map

from sunkit_image.reproject import carrington_header
from sunkit_image.tests.helpers import figure_test


@figure_test
def test_reproject(image_remote):
    aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

    car_header = carrington_header(aia_map, shape_out=[360, 720], projection_code="CAR")
    car_map = aia_map.reproject_to(car_header)

    cea_header = carrington_header(aia_map, shape_out=[90, 180], projection_code="CEA")
    cea_map = aia_map.reproject_to(cea_header)

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection=car_map)
    car_map.plot()

    ax2 = fig.add_subplot(122, projection=cea_map)
    cea_map.plot()
