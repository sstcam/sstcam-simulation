from sstcam_simulation.data import get_data
from os.path import exists
import numpy as np
import pandas as pd
from numba import njit
from astropy import units as u

NSB_FLUX_UNIT = 1/(u.cm**2 * u.ns * u.sr)
NSB_DIFF_FLUX_UNIT = NSB_FLUX_UNIT / u.nm


@njit(fastmath=True)
def _pixel_active_solid_angle_nb(pixel_diameter, focal_length):
    pixel_area = pixel_diameter**2
    equivalent_circular_area = pixel_diameter**2 / 4 * np.pi
    angle_pixel = pixel_diameter / focal_length
    area_ratio = pixel_area / equivalent_circular_area
    return 2 * np.pi * (1.0 - np.cos(0.5 * angle_pixel)) * area_ratio


@njit(fastmath=True)
def _integrate(wavelength, nsb_diff_flux, wavelength_min, wavelength_max):
    within = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
    nsb_within = nsb_diff_flux[within]
    return np.sum(nsb_within) - 0.5 * (nsb_within[0] + nsb_within[-1])


class CameraEfficiency:
    def __init__(self, path=get_data("datasheet/p4eff_ASTRI-CHEC.lis")):
        """
        Calculate parameters related to the camera efficiency

        Formulae and data obtained from the excel files at:
        https://www.mpi-hd.mpg.de/hfm/CTA/MC/Prod4/Config/Efficiencies
        Credit: Konrad Bernloehr

        Parameters
        ----------
        path : str
            Path to the efficiency file from the website - needs to be downloaded first
        """
        if not exists(path):
            raise ValueError(f"No file found at {path}, have you downloaded the file?")

        columns = [
            "wavelength",
            "eff",
            "eff+atm.trans.",
            "q.e.",
            "ref.",
            "masts",
            "filter",
            "funnel",
            "atm.trans.",
            "Ch.light",
            "NSB",
            "atm.corr.",
            "NSB site",
            "NSB site*eff",
            "NSB B&E",
            "NSB B&E*eff",
        ]
        self._df = pd.read_csv(path, delimiter=r"\s+", names=columns)

        #TODO: pass via arguments and interp

        self.wavelength = u.Quantity(self._df['wavelength'], 'nm')
        self._nsb_diff_flux = u.Quantity(self._df['NSB site'], '10^9 / (nm s m^2 sr)')
        self._atmospheric_transmissivity = u.Quantity(self._df['atm.trans.'].values, '1/nm')

        self.telescope_transmissivity = self._df['masts'].values[0]
        self.mirror_reflectivity = self._df['ref.'].values
        self.window_transmissivity = self._df['filter'].values
        self.pde = self._df['q.e.'].values
        self.cherenkov_scale = 1

        self.mirror_area = u.Quantity(7.931, 'm2')
        self.pixel_diameter = u.Quantity(0.0062, 'm')
        self.focal_length = u.Quantity(2.152, 'm/radian')

        self._nsb_flux_300_650 = self._integrate_nsb(
            self._nsb_diff_flux_on_ground, u.Quantity(300, u.nm), u.Quantity(650, u.nm)
        )

    @property
    def _pixel_active_solid_angle(self):
        solid_angle = _pixel_active_solid_angle_nb(
            self.pixel_diameter.to_value(u.m),
            self.focal_length.to_value(u.m/u.radian)
        )
        return u.Quantity(solid_angle, u.sr)

    @u.quantity_input
    def scale_pde(self, wavelength: u.nm, pde_at_wavelength):
        self.pde *= pde_at_wavelength / np.interp(wavelength, self.wavelength, self.pde)

    @u.quantity_input
    def _integrate_nsb(self, nsb_diff_flux, wavelength_min: u.nm, wavelength_max: u.nm):
        integral = _integrate(
            self.wavelength.to_value(u.nm),
            nsb_diff_flux.to_value(NSB_DIFF_FLUX_UNIT),
            wavelength_min.to_value(u.nm),
            wavelength_max.to_value(u.nm),
        )
        return u.Quantity(integral, NSB_FLUX_UNIT)

    @property
    def _nsb_diff_flux_on_ground(self):
        return self._nsb_diff_flux

    @property
    def _nsb_diff_flux_at_camera(self):
        f = self.telescope_transmissivity * self.mirror_reflectivity
        return self._nsb_diff_flux_on_ground * f

    @property
    def _nsb_diff_flux_at_pixel(self):
        return self._nsb_diff_flux_at_camera * self.window_transmissivity

    @property
    def _nsb_diff_flux_inside_pixel(self):
        return self._nsb_diff_flux_at_pixel * self.pde

    @property
    def _nsb_flux_inside_pixel(self):
        wl_min = u.Quantity(200, u.nm)
        wl_max = u.Quantity(999, u.nm)
        return self._integrate_nsb(self._nsb_diff_flux_inside_pixel, wl_min, wl_max)

    @property
    def _nsb_rate_inside_pixel(self):
        rate = self._nsb_flux_inside_pixel * self._pixel_active_solid_angle * self.mirror_area
        return rate

    @u.quantity_input
    def get_scaled_nsb_rate(self, nsb_flux: NSB_FLUX_UNIT):
        scale = nsb_flux / self._nsb_flux_300_650
        return self._nsb_rate_inside_pixel * scale

    @property
    def nominal_nsb_rate(self):
        return self.get_scaled_nsb_rate(u.Quantity(0.24, NSB_FLUX_UNIT))

    @property
    def high_nsb_rate(self):
        return self.get_scaled_nsb_rate(u.Quantity(4.3, NSB_FLUX_UNIT))

    @u.quantity_input
    def _integrate_cherenkov(self, cherenkov_diff_flux, wavelength_min: u.nm, wavelength_max: u.nm):
        integral = _integrate(
            self.wavelength.to_value(u.nm),
            cherenkov_diff_flux.to_value(1 / u.nm),
            wavelength_min.to_value(u.nm),
            wavelength_max.to_value(u.nm),
        )
        return integral

    @property
    def _cherenkov_diff_flux_on_ground(self):
        ref_wavelength = u.Quantity(400, u.nm)
        return (ref_wavelength / self.wavelength)**2 * self._atmospheric_transmissivity * self.cherenkov_scale

    @property
    def _cherenkov_diff_flux_at_camera(self):
        f = self.telescope_transmissivity * self.mirror_reflectivity
        return self._cherenkov_diff_flux_on_ground * f

    @property
    def _cherenkov_diff_flux_at_pixel(self):
        return self._cherenkov_diff_flux_at_camera * self.window_transmissivity

    @property
    def _cherenkov_diff_flux_inside_pixel(self):
        return self._cherenkov_diff_flux_at_pixel * self.pde

    @property
    def effective_cherenkov_pde(self):
        flux = self._cherenkov_diff_flux_at_pixel
        wl_min = u.Quantity(200, u.nm)
        wl_max = u.Quantity(999, u.nm)
        n_photons = self._integrate_cherenkov(flux, wl_min, wl_max)
        n_pe = self._integrate_cherenkov(flux * self.pde, wl_min, wl_max)
        return n_pe / n_photons
