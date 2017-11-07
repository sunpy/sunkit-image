from __future__ import print_function, division

import numpy as np

import astropy.units as u

import sunpy.map


def find_pixel_radii(m, scale=None):
    """
    Find the distance of every pixel in a map from the center of the Sun.
    The answer is returned in units of solar radii.

    Parameters
    ----------
    m :
        A sunpy map object

    scale :


    Returns
    -------

    """
    x, y = np.meshgrid(*[np.arange(v.value) for v in m.dimensions]) * u.pix
    hpc_coords = m.pixel_to_data(x, y)
    radii = np.sqrt(hpc_coords.Tx ** 2 + hpc_coords.Ty ** 2)
    if scale is None:
        return u.R_sun * (radii / m.rsun_obs)
    else:
        return u.R_sun * (radii / scale)


def fit_radial_intensity(m,
                         rmin=1.0*u.R_sun,
                         fit_rmax=1.5*u.R_sun,
                         rsun_step_size=0.01*u.R_sun,
                         degree=1,
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
    r = find_pixel_radii(m).to(u.R_sun).value
    rsun_array = np.arange(rmin.to(u.R_sun).value, r.max(), rsun_step_size.to(u.R_sun).value)
    radial_emission = np.array([annular_function(m.data[(r > this_r) * (r < this_r + rsun_step_size)], **annular_function_kwargs)
                  for this_r in rsun_array])

    return np.polyfit(rsun_array[rsun_array < fit_rmax],
                        np.log(radial_emission[rsun_array < fit_rmax]), degree)


class OffLimbEnhance(object):
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

        # Get the radii
        self._r = find_pixel_radii(self.smap, scale=self.scale).to(u.Rsun).value

        # Get the fit polynomial
        self.fit_polynomial = fit_radial_intensity(self.smap, fit_range=self.fit_range,
                                                   step_size=self.step_size,
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

    def scale_factor(self, normalization_radius=1*u.Rsun, normalization_factor=1):
        """

        Returns
        -------

        """
        nr = normalization_radius.to(u.Rsun).value

        scaled_polynomial = np.poly1d(self.fit_polynomial)(self._r) - np.poly1d(self.fit_polynomial)(nr)
        factor = np.exp(scaled_polynomial)

        #
        factor[self._r < nr] = normalization_factor

        # On disk, the scaling factor is 1
        factor[self._r < 1] = 1
        return factor

    def enhanced_map(self, normalization_radius=1*u.Rsun):
        return sunpy.map.Map(self.smap.data * self.scale_factor(normalization_radius=normalization_radius), self.smap.meta)