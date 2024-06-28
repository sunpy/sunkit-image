import matplotlib.pyplot as plt
import numpy as np
import pytest
import sunpy.data.sample
import sunpy.map

import sunkit_image.enhance as enhance
from sunkit_image.tests.helpers import figure_test


@figure_test
@pytest.mark.remote_data()
def test_mgn(aia_171):
    out = enhance.mgn(aia_171)
    assert type(out) == type(aia_171)
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


def test_multiscale_gaussian(map_test):
    result1 = np.zeros((4, 4), dtype=float)
    expect3 = enhance.mgn(map_test, sigma=[1])
    assert np.allclose(result1, expect3)


def test_nans_raise_warning(map_test):
    map_test[0, 0] = np.nan
    with pytest.warns(UserWarning, match="One or more entries in the input data are NaN."):
        enhance.mgn(map_test)


@figure_test
@pytest.mark.remote_data()
def test_wow(aia_171):
    out = enhance.wow(aia_171)
    assert type(out) == type(aia_171)
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
