from __future__ import print_function, division

import numpy as np

import astropy.units as u

import sunpy.map

from sunkit_image.utils.utils import find_pixel_radii


def fit_radial_intensity(m, r, degree=1,
                         annular_function=np.mean,
                         **annular_function_kwargs):
    """
    Fits a polynomial function to the natural logarithm of an estimate of the
    intensity as a function of radius.


    Parameters
    ----------
    m
    rmin
    fit_rmax
    rsun_step_size
    degree
    annular_function
    annular_function_kwargs

    Returns
    -------

    """
    map_r = find_pixel_radii(m).to(u.R_sun).value
    radial_emission = np.zeros(shape=r.size-1)
    for i in range(0, r.size):
        lower_radius = r[i]
        upper_radius = r[i+1]
        radial_emission[i] = annular_function(m.data[(map_r > lower_radius) * (map_r < upper_radius)], **annular_function_kwargs)

    return np.polyfit(r,
                        np.log(radial_emission[rsun_array < fit_rmax]), degree)


class FitRadialIntensity:
    def __init__(self, m, r, degree=1,
                         annular_function=np.mean,
                         **annular_function_kwargs):



class OffLimbIntensity(object):
    def __init__(self, smap, fit_range=[1.0, 1.5]*u.R_sun, step_size=0.01*u.R_sun,
                 degree=1, scale=None, annular_function=np.mean, **annular_function_kwargs):
        """

        Parameters
        ----------
        smap
        fit_range
        step_size
        degree
        scale
        annular_function
        annular_function_kwargs
        """
        self.smap = smap
        self.fit_range = fit_range
        self.step_size = step_size
        self.degree = degree
        self.scale = scale
        self.annular_function = annular_function
        self.annular_function_kwargs = annular_function_kwargs

        # Get the radii for every pixel
        self._map_r = find_pixel_radii(self.smap, scale=self.scale).to(u.R_sun).value

        # Get the
        self.r = np.arange(self.fit_range[0].to(u.R_sun).value,
                           self.fit_range[1].to(u.R_sun).value,
                           self.step_size.to(u.R_sun).value)

        # Get the fit polynomial
        self.fit_polynomial = fit_radial_intensity(self.smap, self.r,
                                                   degree=self.degree,
                                                   annular_function=self.annular_function,
                                                   **self.annular_function_kwargs)

    def best_fit(self):
        """
        The best fit to the log(intensity) as a function of radius

        Returns
        -------

        """
        return np.poly1d(self.fit_polynomial)

    def compensation(self, normalization_radius=1*u.R_sun, normalization_factor=1):
        """

        Returns
        -------

        """
        nr = normalization_radius.to(u.R_sun).value

        scaled_polynomial = np.poly1d(self.fit_polynomial)(self._map_r) - np.poly1d(self.fit_polynomial)(nr)
        factor = np.exp(scaled_polynomial)

        #
        factor[self._map_r < nr] = normalization_factor

        # On disk, the scaling factor is 1
        factor[self._map_r < 1] = 1
        return factor

    def compensate(self, normalization_radius=1*u.R_sun)
        return sunpy.map.Map(self.smap.data * self.compensation(normalization_radius=normalization_radius), self.smap.meta)