import numpy as np
from skimage import data
import sys
sys.path.append("..noise")

from noise import *

img = data.camera()


def convmtx2_test():

    tt = convmtx2(img, 11.0, 7.0)
    assert tt.shape == ((11 - img.shape[0] + 1) * (7 - img.shape[1] + 1), 11.0 * 7.0)


def noiselevel_test():

    noise_levels = np.array([5.0, 10.0, 20.0, 42.0])
    n_levels = np.zeros_like(noise_levels)
    n_patches = np.zeros_like(noise_levels)

    for n in range(noise_levels.size):
        noise = img + np.random.standard_normal(img.shape) * noise_levels[n]
        output = initialise(noise, patchsize=11, itr=5)
        n_levels[n] = output.nlevel
        n_patches[n] = output.num

    assert np.abs(1 - n_levels / noise_levels) < 0.1
    assert n_patches > 10000.0


def weaktexturemask_test():
    noise_levels = 5
    noise = img + np.random.standard_normal(img.shape) * noise_levels
    output = initialise(noise, patchsize=11, itr=5)

    assert np.sum(output.mask) / output.mask.size < 1.0
