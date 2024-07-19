import matplotlib.pyplot as plt
import pytest

import sunpy.data.sample
import sunpy.map

import sunkit_image.enhance as enhance
from sunkit_image.tests.helpers import figure_test


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


@figure_test
def test_mgn_cutout(aia_171_cutout):
    out = enhance.mgn(aia_171_cutout)
    assert type(out) == type(aia_171_cutout)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=out)
    out.plot(axes=ax)
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
def test_wow_cutout(aia_171_cutout):
    out = enhance.wow(aia_171_cutout)
    assert type(out) == type(aia_171_cutout)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=out)
    out.plot(axes=ax)
    return fig
