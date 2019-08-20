import numpy as np
import pytest
from skimage import data

import sunkit_image.utils.noise as nf


@pytest.fixture
def img():
    image = data.camera()
    return image


def test_conv2d_matrix_size(img):

    tt = nf.conv2d_matrix(img, 11.0, 7.0)
    assert tt.shape == ((11 - img.shape[0] + 1) * (7 - img.shape[1] + 1), 11.0 * 7.0)


def test_noiselevel(img):

    noise_levels = np.array([5.0, 10.0, 20.0, 42.0])
    n_levels = np.zeros_like(noise_levels)
    n_patches = np.zeros_like(noise_levels)

    for n in range(noise_levels.size):
        noise = img + np.random.standard_normal(img.shape) * noise_levels[n]
        output = nf.noise_estimation(noise, patchsize=11, iterations=5)
        n_levels[n] = output["nlevel"]
        n_patches[n] = output["num"]

    assert np.abs(1 - n_levels.all() / noise_levels.all()) < 0.1
    assert all(n_patches > 10000.0)


def test_weak_texture_mask(img):

    noise_levels = 5
    noise = img + np.random.standard_normal(img.shape) * noise_levels
    output = nf.noise_estimation(noise, patchsize=11, iterations=5)

    assert np.sum(output["mask"]) / output["mask"].size < 1.0
