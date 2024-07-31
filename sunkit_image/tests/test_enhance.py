import matplotlib.pyplot as plt
import numpy as np
import pytest

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy.map

import sunkit_image.enhance as enhance
from sunkit_image.tests.helpers import figure_test

pytestmark = [pytest.mark.filterwarnings("ignore:Missing metadata for observer"), pytest.mark.filterwarnings("ignore:Missing metadata for observation time")]


@figure_test
@pytest.mark.remote_data()
def test_mgn(aia_171):
    out = enhance.mgn(aia_171)
    if isinstance(out, sunpy.map.GenericMap):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=out)
        out.plot(axes=ax)
        return fig
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(out, origin="lower", interpolation="nearest", cmap="sdoaia171")
    return fig


@pytest.fixture()
def map_test():
    return np.ones((4, 4), dtype=float)


def test_nans_raise_warning(map_test):
    map_test[0, 0] = np.nan
    with pytest.warns(UserWarning, match="One or more entries in the input data are NaN."):
        enhance.mgn(map_test)


@figure_test
@pytest.mark.remote_data()
def test_mgn_submap(aia_171_map):
    top_right = SkyCoord(0 * u.arcsec, -200 * u.arcsec, frame=aia_171_map.coordinate_frame)
    bottom_left = SkyCoord(-900 * u.arcsec, -900 * u.arcsec, frame=aia_171_map.coordinate_frame)
    aia_171_map_submap = aia_171_map.submap(bottom_left, top_right=top_right)
    out = enhance.mgn(aia_171_map_submap)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=out)
    out.plot(axes=ax)
    return fig


@figure_test
def test_mgn_cutout(aia_171_cutout):
    out = enhance.mgn(aia_171_cutout)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=out)
    out.plot(axes=ax, clip_interval=(1, 99) * u.percent)
    return fig


@figure_test
@pytest.mark.remote_data()
def test_wow(aia_171):
    out = enhance.wow(aia_171)
    if isinstance(out, sunpy.map.GenericMap):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=out)
        out.plot(axes=ax)
        return fig
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(out, origin="lower", interpolation="nearest", cmap="sdoaia171")
    return fig


@figure_test
@pytest.mark.remote_data()
def test_wow_submap(aia_171_map):
    top_right = SkyCoord(0 * u.arcsec, -200 * u.arcsec, frame=aia_171_map.coordinate_frame)
    bottom_left = SkyCoord(-900 * u.arcsec, -900 * u.arcsec, frame=aia_171_map.coordinate_frame)
    aia_171_map_submap = aia_171_map.submap(bottom_left, top_right=top_right)
    out = enhance.wow(aia_171_map_submap)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=out)
    out.plot(axes=ax)
    return fig


@figure_test
def test_wow_cutout(aia_171_cutout):
    out = enhance.wow(aia_171_cutout)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=out)
    out.plot(axes=ax)
    return fig
