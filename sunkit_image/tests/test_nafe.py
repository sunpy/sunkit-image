import numpy as np
import pytest

import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE

import sunkit_image.nafe as NAFE


def test_get_membership_grade():
    expected = np.array([[0.65289406, 0.73715583, 0.77417820, 0.73715583, 0.65289406],
                         [0.73715583, 0.86813217, 0.92826105, 0.86813217, 0.73715583],
                         [0.77417820, 0.92826105, 1.,         0.92826105, 0.77417820],
                         [0.73715583, 0.86813217, 0.92826105, 0.86813217, 0.73715583],
                         [0.65289406, 0.73715583, 0.77417820, 0.73715583, 0.65289406]])
    assert(np.allclose(NAFE._get_membership_grade(5), expected))


def test_get_transform():
    assert(NAFE._transform(in_margin=(0, 5), out_margin=(3, 5), old_value=2.5) == 4)
    assert(NAFE._transform(in_margin=(0, 5), out_margin=(3, 5), old_value=2.5, power=1/2) == 3.5)


def test_nafe_args_checks():
    in_map = sunpy.map.Map(AIA_171_IMAGE)

    with pytest.raises(ValueError):
        NAFE.nafe(in_map, gamma=-1)

    with pytest.raises(ValueError):
        NAFE.nafe(in_map, nafe_weight=-1)
    with pytest.raises(ValueError):
        NAFE.nafe(in_map, nafe_weight=2)

    with pytest.raises(ValueError):
        NAFE.nafe(in_map, hist_bins=-1)
    with pytest.raises(ValueError):
        NAFE.nafe(in_map, hist_bins=0)

    with pytest.raises(ValueError):
        NAFE.nafe(in_map, noise_reduction_sigma=-1)

    with pytest.raises(ValueError):
        NAFE.nafe(in_map, n=-1)
    with pytest.raises(ValueError):
        NAFE.nafe(in_map, n=0)
    with pytest.raises(ValueError):
        NAFE.nafe(in_map, n=1.5)

    with pytest.raises(ValueError):
        NAFE.nafe(in_map, nproc=-1)
    with pytest.raises(ValueError):
        NAFE.nafe(in_map, nproc=0)
    with pytest.raises(ValueError):
        NAFE.nafe(in_map, nproc=1.5)
    with pytest.raises(ValueError):
        NAFE.nafe(in_map, nproc=10000)
