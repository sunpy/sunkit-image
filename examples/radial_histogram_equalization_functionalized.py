"""
=============================
Radial Histogram Equalization
=============================

This example applies the Radial Histogram Equalizing Filter (`sunkit_image.radial.rhef`) to a sunpy map.
"""

import logging

import astropy.units as u
import matplotlib.pyplot as plt
import sunpy.data.sample
import sunpy.map
from astropy.coordinates import SkyCoord

import sunkit_image.enhance as enhance
import sunkit_image.radial as radial
from sunkit_image.utils import equally_spaced_bins

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_sample_data():
    """
    Load the SunPy sample data AIA image.
    """
    logging.info("Loading sample data...")
    return sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE, autoalign=True)


def plot_original_and_filtered_map(aia_map, rhef_map):
    """
    Plot the original and RHE filtered SunPy maps side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex="all", sharey="all", subplot_kw={"projection": aia_map})

    aia_map.plot(axes=axes[0], clip_interval=(1, 99.99) * u.percent)
    axes[0].set_title("Original AIA Map")

    rhef_map.plot(axes=axes[1])
    axes[1].set_title(r"RHE Filtered Map, $\Upsilon$=0.35")

    fig.tight_layout()


def apply_rhef(aia_map):
    """
    Apply the Radial Histogram Equalizing Filter (RHEF) to the SunPy map.
    """
    logging.info("Performing default RHE...")
    radial_bin_edges = equally_spaced_bins(0, 2, aia_map.data.shape[0] // 2)
    radial_bin_edges *= u.R_sun
    return radial.rhef(aia_map, radial_bin_edges)


def plot_upsilon_examples(aia_map, upsilon_list):
    """
    Plot examples of different upsilon values applied to the SunPy map.
    """
    logging.info("Plotting Upsilon Examples...")
    top_right = SkyCoord(1200 * u.arcsec, 0 * u.arcsec, frame=aia_map.coordinate_frame)
    bottom_left = SkyCoord(0 * u.arcsec, -1200 * u.arcsec, frame=aia_map.coordinate_frame)
    aia_map_cropped = aia_map.submap(bottom_left, top_right=top_right)

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 10), sharex="all", sharey="all", subplot_kw={"projection": aia_map_cropped}
    )
    axes = axes.flatten()

    aia_map_cropped.plot(axes=axes[0], clip_interval=(1, 99.99) * u.percent)
    axes[0].set_title("Original AIA Map")

    for i, upsilon in enumerate(upsilon_list):
        out_map = radial.rhef(aia_map, upsilon=upsilon, method="scipy")
        out_map_crop = out_map.submap(bottom_left, top_right=top_right)
        out_map_crop.plot(axes=axes[i + 1])
        axes[i + 1].set_title(f"Upsilon = {upsilon}")
        lgg = "Plotted Upsilon = " + str(upsilon)
        logging.info(lgg)

    fig.tight_layout()


def plot_serial_processing(aia_map):
    """
    Apply multiple filters in a row and plot the results.
    """
    logging.info("Plotting Serial Processing...")

    mgn_map = enhance.mgn(aia_map)
    wow_map = enhance.wow(aia_map)

    rhef_map = radial.rhef(aia_map)
    rhef_mgn_map = radial.rhef(mgn_map)
    rhef_wow_map = radial.rhef(wow_map)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex="all", sharey="all", subplot_kw={"projection": aia_map})
    axes = axes.flatten()

    rhef_map.plot(axes=axes[0])
    axes[0].set_title("RHEF(smap)")

    mgn_map.plot(axes=axes[1], norm=None)
    axes[1].set_title("MGN(smap)")

    wow_map.plot(axes=axes[2], norm=None)
    axes[2].set_title("WOW(smap)")

    toplot = (rhef_map.data + rhef_mgn_map.data) / 2
    combo_map = sunpy.map.Map(toplot, rhef_map.meta)
    combo_map.plot_settings["norm"] = None
    combo_map.plot(axes=axes[3])
    axes[3].set_title("AVG( RHEF(smap), RHEF(MGN(smap) )")

    rhef_mgn_map.plot(axes=axes[4])
    axes[4].set_title("RHEF( MGN(smap) )")

    rhef_wow_map.plot(axes=axes[5])
    axes[5].set_title("RHEF( WOW(smap) )")

    fig.tight_layout()
    logging.info("Done!")


def main():
    aia_map = load_sample_data()
    rhef_map = apply_rhef(aia_map)
    plot_original_and_filtered_map(aia_map, rhef_map)

    upsilon_list = [0.35, None, (0.1, 0.1), (0.5, 0.5), (0.8, 0.8)]
    plot_upsilon_examples(aia_map, upsilon_list)

    plot_serial_processing(aia_map)

    plt.show()


if __name__ == "__main__":
    main()
