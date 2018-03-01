#
# Tests for the utilities
#

from __future__ import absolute_import, division, print_function

import pytest

import numpy as np

from sunkit_image.utils.utils import _equally_spaced_bins, bin_edge_summary


def test_equally_spaced_bins():
    # test the default
    esb = _equally_spaced_bins()
    assert esb.shape == (2, 100)
    assert esb[0, 0] == 1.0
    assert esb[1, 0] == 1.01
    assert esb[0, 99] == 1.99
    assert esb[1, 99] == 2.00

    # Bins are 0.015 wide
    esb2 = _equally_spaced_bins(inner_value=0.5)
    assert esb2.shape == (2, 100)
    assert esb2[0, 0] == 0.5
    assert esb2[1, 0] == 0.515
    assert esb2[0, 99] == 1.985
    assert esb2[1, 99] == 2.00

    # Bins are 0.2 wide
    esb2 = _equally_spaced_bins(outer_value=3.0)
    assert esb2.shape == (2, 100)
    assert esb2[0, 0] == 1.0
    assert esb2[1, 0] == 1.02
    assert esb2[0, 99] == 2.98
    assert esb2[1, 99] == 3.00

    # Bins are 0.01 wide
    esb2 = _equally_spaced_bins(nbins=1000)
    assert esb2.shape == (2, 1000)
    assert esb2[0, 0] == 1.0
    assert esb2[1, 0] == 1.001
    assert esb2[0, 999] == 1.999
    assert esb2[1, 999] == 2.000

    # The radii have the correct relative sizes
    with pytest.raises(ValueError):
        _equally_spaced_bins(inner_value=1.0, outer_value=1.0)
    with pytest.raises(ValueError):
        _equally_spaced_bins(inner_value=1.5, outer_value=1.0)

    # The number of bins is strictly greater than 0
    with pytest.raises(ValueError):
        _equally_spaced_bins(nbins=0)


def test_bin_edge_summary():
    esb = _equally_spaced_bins()

    center = bin_edge_summary(esb, 'center')
    assert center.shape == (100,)
    assert center[0] == 1.005
    assert center[99] == 1.995

    left = bin_edge_summary(esb, 'left')
    assert left.shape == (100,)
    assert left[0] == 1.0
    assert left[99] == 1.99

    right = bin_edge_summary(esb, 'right')
    assert right.shape == (100,)
    assert right[0] == 1.01
    assert right[99] == 2.0

    # Correct selection of summary type
    with pytest.raises(ValueError):
        bin_edge_summary(esb, 'should raise the error')

    # The correct shape of bin edges are passed in
    with pytest.raises(ValueError):
        bin_edge_summary(np.arange(0, 10), 'center')
    with pytest.raises(ValueError):
        bin_edge_summary(np.zeros((3, 4)), 'center')
