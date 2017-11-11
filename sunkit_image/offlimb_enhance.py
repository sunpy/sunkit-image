from __future__ import print_function, division

import numpy as np

import astropy.units as u

import sunpy.map

from sunkit_image.utils.utils import find_pixel_radii


def bin_edge_summary(r, binfit='center'):
    """

    """
    if binfit == 'center':
        rfit = 0.5*(r[0:-1] + r[1:])
    elif binfit == 'left':
        rfit = r[0:-1]
    elif binfit == 'right':
        rfit = r[1:]
    else:
        raise ValueError('Keyword "binfit" must have value "center", "left" or "right"')
    return rfit


def fit_radial_intensity(smap, r, fit_range=[1.0, 1.5]*u.R_sun, degree=1, binfit='center',
                         summary=np.mean,
                         **summary_kwargs):
    """
    Fits a polynomial function to the natural logarithm of an estimate of the
    intensity as a function of radius.

    Parameters
    ----------


    Returns
    -------

    """

    # Get the emission as a function of intensity.
    radial_emission = get_intensity_summary(smap, r, summary=summary, **summary_kwargs)

    # The radial emission is found by calculating a summary statistic of the
    # intensity in bins with bin edges defined by the input 'r'. If 'r' has N
    # values, the radial emission returned has N-1 values.  In order to perform
    # the fit a summary of the bins must be calculated.
    rfit = bin_edge_summary(r, binfit=binfit)

    # Calculate which radii are to be fit.
    fit_here = fit_range[0] < rfit < fit_range[1]

    # Return the values of a polynomial fit to the log of the radial emission
    return np.polyfit(rfit[fit_here], np.log(radial_emission[fit_here]), degree)


def get_intensity_summary(smap, r, scale=None, summary=np.mean, **summary_kwargs):
    """
    Get a summary statistic of the intensity in a map as a function of radius.

    Parameters
    ----------
    smap : sunpy.map.Map
        A sunpy map.

    r : `~astropy.units.Quantity`
        A one-dimensional array of bin edges.

    scale : `~astropy.units.Quantity`
        The radius of the Sun expressed in map units.  For example, in typical
        helioprojective Cartesian maps the solar radius is expressed in units
        of arcseconds.

    Returns
    -------
    intensity summary : `~numpy.array`
        A summary statistic of the radial intensity in the bins defined by the
        bin edges.  If "r" has N bins, the returned array has N-1 values.
    """
    # Get the radial distance of every pixel from the center of the Sun.
    map_r = find_pixel_radii(smap, scale=scale).to(u.R_sun).value

    # Calculate the summary statistic in the radial bins.
    return np.asarray([summary(smap.data[(map_r > r[i]) * (map_r < r[i+1])], **summary_kwargs) for i in range(0, r.size-1)])



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