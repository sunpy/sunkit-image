import numpy as np
import pytest
import astropy.units as u
import sunpy.map
from sunpy.data.sample import AIA_171_IMAGE
from sunkit_image.radial import intensity_enhance, nrgf, fnrgf, rhef
from sunkit_image.utils import equally_spaced_bins

# Load a sample SunPy map for testing
smap = sunpy.map.Map(AIA_171_IMAGE)

# Define common parameters for testing
radial_bin_edges = equally_spaced_bins(0, 2, 10) * u.R_sun

def test_intensity_enhance():
    enhanced_map = intensity_enhance(smap, radial_bin_edges)
    assert isinstance(enhanced_map, sunpy.map.GenericMap)
    assert enhanced_map.data.shape == smap.data.shape

def test_nrgf():
    nrgf_map = nrgf(smap, radial_bin_edges)
    assert isinstance(nrgf_map, sunpy.map.GenericMap)
    assert nrgf_map.data.shape == smap.data.shape

def test_fnrgf():
    order = 3
    mean_attenuation_range = (1.0, 0.0)
    std_attenuation_range = (1.0, 0.0)
    cutoff = 0
    ratio_mix = (15, 1)
    fnrgf_map = fnrgf(smap, radial_bin_edges, order, mean_attenuation_range, std_attenuation_range, cutoff, ratio_mix)
    assert isinstance(fnrgf_map, sunpy.map.GenericMap)
    assert fnrgf_map.data.shape == smap.data.shape

def test_rhef():
    rhef_map = rhef(smap, radial_bin_edges)
    assert isinstance(rhef_map, sunpy.map.GenericMap)
    assert rhef_map.data.shape == smap.data.shape
