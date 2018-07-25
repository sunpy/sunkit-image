import numpy as np
from skimage import data
from NoiseLevel import NoiseLevelEstimation as nle


img = data.camera()


def convmtx2_test():

    tt = nle.convmtx2(img, 11., 7.)
    assert tt.shape == ((11-img.shape[0] + 1) * (7 - img.shape[1] + 1),11.*7.)


def noiselevel_test():

    noise_levels = np.array([5., 10., 20., 42.])
    n_levels = np.zeros_like(noise_levels)
    n_patches = np.zeros_like(noise_levels)

    for n in range(noise_levels.size):
        noise = img + np.random.standard_normal(img.shape) * noise_levels[n]
        output = nle(noise, patchsize=11, itr=5)
        n_levels[n] = output.nlevel
        n_patches[n] = output.num

    assert np.abs(1-n_levels/noise_levels) < 0.1
    assert n_patches > 10000.


def  weaktexturemask_test():
    noise_levels = 5
    noise = img + np.random.standard_normal(img.shape) * noise_levels
    output = nle(noise, patchsize=11, itr=5)

    assert np.sum(output.mask)/output.mask.size < 1.0
