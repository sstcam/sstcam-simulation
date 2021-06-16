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
    # print(1/area_ratio)
    # raise ValueError
    return 2 * np.pi * (1.0 - np.cos(0.5 * angle_pixel)) * area_ratio


@njit(fastmath=True)
def _integrate(wavelength, nsb_diff_flux, wavelength_min, wavelength_max):
    within = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
    nsb_within = nsb_diff_flux[within]
    return np.sum(nsb_within) - 0.5 * (nsb_within[0] + nsb_within[-1])


class CameraEfficiency:
    def __init__(
            self,
            path_window=get_data("datasheet/efficiency/prod4_window.csv"),
            path_pde=get_data("datasheet/efficiency/prod4_pde.csv")
    ):
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
        if not exists(path_window):
            raise ValueError(f"No file found at {path_window}")

        if not exists(path_pde):
            raise ValueError(f"No file found at {path_pde}")

        wavelength = np.arange(200, 999)

        # Read environment arrays
        df_env = pd.read_csv(get_data("datasheet/efficiency/environment.csv"))
        env_wavelength = df_env['wavelength'].values

        def interp_env(y):
            return np.interp(wavelength, env_wavelength, y)

        nsb_diff_flux = interp_env(df_env['nsb_site'].values)
        self._nsb_diff_flux = u.Quantity(nsb_diff_flux, '10^9 / (nm s m^2 sr)')
        moonlight_diff_flux = interp_env(df_env['moonlight'].values)
        # TODO: normalise moonlight
        self._moonlight_diff_flux = u.Quantity(moonlight_diff_flux, '10^9 / (nm s m^2 sr)')
        atmospheric_transmissivity = interp_env(df_env['atmospheric_transmissivity'].values)
        self._atmospheric_transmissivity = u.Quantity(atmospheric_transmissivity, '1/nm')

        # Read telescope arrays
        df_tel = pd.read_csv(get_data("datasheet/efficiency/prod4_astri_telescope.csv"))
        tel_wavelength = df_tel['wavelength'].values

        def interp_tel(y):
            return np.interp(wavelength, tel_wavelength, y)

        self.telescope_transmissivity = interp_tel(df_tel['telescope_transmissivity'].values)
        self.mirror_reflectivity = interp_tel(df_tel['mirror_reflectivity'].values)

        # Read camera window transmissivity
        df_window = pd.read_csv(path_window)
        window_wavelength = df_window["wavelength"].values
        window_transmissivity = df_window['window_transmissivity'].values
        self.window_transmissivity = np.interp(wavelength, window_wavelength, window_transmissivity)

        # Read camera pde
        df_pde = pd.read_csv(path_pde)
        pde_wavelength = df_pde["wavelength"].values
        pde = df_pde['pde'].values
        self.pde = np.interp(wavelength, pde_wavelength, pde)

        self.mirror_area = u.Quantity(7.931, 'm2')
        self.pixel_diameter = u.Quantity(0.0062, 'm')
        self.focal_length = u.Quantity(2.152, 'm/radian')

        self.wavelength = u.Quantity(wavelength, 'nm')

        self._nsb_flux_300_650 = self._integrate_nsb(
            self._nsb_diff_flux_on_ground, u.Quantity(300, u.nm), u.Quantity(650, u.nm)
        )

        self._moonlight_flux_300_650 = self._integrate_nsb(
            self._moonlight_diff_flux_on_ground, u.Quantity(300, u.nm), u.Quantity(650, u.nm)
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

    @property
    def _moonlight_diff_flux_on_ground(self):
        return self._moonlight_diff_flux

    @property
    def _moonlight_diff_flux_at_camera(self):
        f = self.telescope_transmissivity * self.mirror_reflectivity
        return self._moonlight_diff_flux_on_ground * f

    @property
    def _moonlight_diff_flux_at_pixel(self):
        return self._moonlight_diff_flux_at_camera * self.window_transmissivity

    @property
    def _moonlight_diff_flux_inside_pixel(self):
        return self._moonlight_diff_flux_at_pixel * self.pde

    @property
    def _moonlight_flux_inside_pixel(self):
        wl_min = u.Quantity(200, u.nm)
        wl_max = u.Quantity(999, u.nm)
        return self._integrate_nsb(self._moonlight_diff_flux_inside_pixel, wl_min, wl_max)

    @property
    def _moonlight_rate_inside_pixel(self):
        rate = self._moonlight_flux_inside_pixel * self._pixel_active_solid_angle * self.mirror_area
        return rate

    @u.quantity_input
    def get_scaled_moonlight_rate(self, moonlight_flux: NSB_FLUX_UNIT):
        scale = moonlight_flux / self._moonlight_flux_300_650
        return self._moonlight_rate_inside_pixel * scale

    @property
    def nominal_moonlight_rate(self):
        # TODO: scaled value
        return self.get_scaled_moonlight_rate(u.Quantity(0.835, NSB_FLUX_UNIT))

    @property
    def high_moonlight_rate(self):
        # TODO: scaled value
        return self.get_scaled_moonlight_rate(u.Quantity(4.3, NSB_FLUX_UNIT))

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
        return (ref_wavelength / self.wavelength)**2 * self._atmospheric_transmissivity

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
