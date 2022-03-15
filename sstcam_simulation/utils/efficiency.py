from sstcam_simulation.data import get_data
from sstcam_simulation.utils.window_durham_needle import WindowDurhamNeedle
from sstcam_simulation.utils.sipm.pde import PDEvsWavelength, read_lct5_resin_coated, read_prototype
import numpy as np
import pandas as pd
from numba import njit
from astropy import units as u
import yaml

NSB_FLUX_UNIT = 1/(u.cm**2 * u.ns * u.sr)
NSB_DIFF_FLUX_UNIT = NSB_FLUX_UNIT / u.nm

PATH_ENV = get_data("datasheet/efficiency/environment.csv")
PROD4_PATH_WINDOW = get_data("datasheet/efficiency/window_prod4.csv")
PROD4_PATH_PDE = get_data("datasheet/efficiency/pde_prod4.csv")
PROD4_PATH_TEL = get_data("datasheet/efficiency/telescope_prod4_astri.csv")
PROD4_PATH_QUAN = get_data("datasheet/efficiency/quantities_prod4.yml")
SSTCAM_PATH_QUAN = get_data("datasheet/efficiency/quantities_sstcam.yml")


@njit(fastmath=True)
def _pixel_active_solid_angle_nb(pixel_diameter, focal_length):
    pixel_area = pixel_diameter**2
    equivalent_circular_area = pixel_diameter**2 / 4 * np.pi
    angle_pixel = pixel_diameter / focal_length
    area_ratio = pixel_area / equivalent_circular_area
    return 2 * np.pi * (1.0 - np.cos(0.5 * angle_pixel)) * area_ratio


@njit(fastmath=True)
def _integrate(x, y, x_min, x_max):
    within = (x >= x_min) & (x <= x_max)
    # x_within = x[within]
    y_within = y[within]
    return np.sum(y_within) - 0.5 * (y_within[0] + y_within[-1])
    # return np.trapz(y_within, x_within)


class CameraEfficiency:
    @u.quantity_input
    def __init__(
            self,
            # scalars
            pixel_diameter: u.m,
            pixel_fill_factor,
            focal_length: u.m/u.radian,
            mirror_area: u.m**2,
            # arrays (vs wavelength)
            telescope_transmissivity,
            mirror_reflectivity,
            window_transmissivity,
            pde,
    ):
        """
        Calculate parameters related to the camera efficiency

        Formulae and data obtained from the excel files at:
        https://www.mpi-hd.mpg.de/hfm/CTA/MC/Prod4/Config/Efficiencies
        Credit: Konrad Bernloehr
        """
        self.wavelength = np.arange(200, 1000) << u.nm
        size = self.wavelength.size
        if not telescope_transmissivity.size == size:
            raise ValueError("All arrays must specify values over full wavelength range")
        if not mirror_reflectivity.size == size:
            raise ValueError("All arrays must specify values over full wavelength range")
        if not window_transmissivity.size == size:
            raise ValueError("All arrays must specify values over full wavelength range")
        if not pde.size == size:
            raise ValueError("All arrays must specify values over full wavelength range")

        self.pixel_diameter = pixel_diameter.to('m')
        self.pixel_fill_factor = pixel_fill_factor
        self.focal_length = focal_length.to("m/radian")
        self.mirror_area = mirror_area.to("m2")
        self.telescope_transmissivity = telescope_transmissivity
        self.mirror_reflectivity = mirror_reflectivity
        self.window_transmissivity = window_transmissivity
        self._pde = pde
        self._pde_scale = 1
        self._cherenkov_scale = 1

        # Read environment arrays
        df_env = pd.read_csv(PATH_ENV)
        diff_flux_unit = u.Unit('10^9 / (nm s m^2 sr)')
        self._nsb_diff_flux = df_env['nsb_site'].values << diff_flux_unit
        self._moonlight_diff_flux = df_env['moonlight'].values << diff_flux_unit
        self._atmospheric_transmissivity = df_env['atmospheric_transmissivity'].values << 1/u.nm

        # Scale cherenkov spectrum to match normalisation
        # scaled to 100 photons/m2 in the wavelength range from 300–600 nm
        # a value typical for γ-ray showers of about 500 GeV viewed at small core distances
        # (from https://jama.cta-observatory.org/perspective.req#/items/28666)
        # TODO: check wrt area
        cherenkov_integral = self._integrate_cherenkov(
            self._cherenkov_diff_flux_on_ground,
            u.Quantity(300, u.nm),
            u.Quantity(600, u.nm)
        )
        self._cherenkov_scale = 100 / cherenkov_integral

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

    @property
    def pde(self):
        return self._pde * self._pde_scale

    @u.quantity_input
    def scale_pde(self, wavelength: u.nm, pde_at_wavelength):
        self._pde_scale = pde_at_wavelength / np.interp(wavelength, self.wavelength, self._pde)

    def reset_pde_scale(self):
        self._pde_scale = 1

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
    def _nsb_diff_flux_at_mirror(self):
        return self._nsb_diff_flux_on_ground * self.telescope_transmissivity

    @property
    def _nsb_diff_flux_at_camera(self):
        return self._nsb_diff_flux_at_mirror * self.mirror_reflectivity

    @property
    def _nsb_diff_flux_at_pixel(self):
        return self._nsb_diff_flux_at_camera * self.window_transmissivity

    @property
    def _nsb_flux_at_pixel(self):
        wl_min = u.Quantity(200, u.nm)
        wl_max = u.Quantity(999, u.nm)
        return self._integrate_nsb(self._nsb_diff_flux_at_pixel, wl_min, wl_max)

    @property
    def _nsb_photon_rate_at_pixel(self):
        rate = self._nsb_flux_at_pixel * self._pixel_active_solid_angle * self.mirror_area
        return rate

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

    @property
    def n_nsb_photoelectrons(self):
        return self._integrate_nsb(
            self._nsb_diff_flux_inside_pixel,
            u.Quantity(200, u.nm),
            u.Quantity(999, u.nm)
        ) * self.pixel_fill_factor

    @property
    def camera_nsb_pde(self):
        n_pe = self.n_nsb_photoelectrons

        # According to Konrad's spreadsheet:
        # normalizing to photons outside sensible range makes no sense
        n_photons = self._integrate_nsb(
            self._nsb_diff_flux_at_camera,
            u.Quantity(300, u.nm),
            u.Quantity(550, u.nm)
        )
        return n_pe / n_photons

    @property
    def telescope_nsb_pde(self):
        n_pe = self.n_nsb_photoelectrons

        # According to Konrad's spreadsheet:
        # normalizing to photons outside sensible range makes no sense
        n_photons = self._integrate_nsb(
            self._nsb_diff_flux_at_mirror,
            u.Quantity(300, u.nm),
            u.Quantity(550, u.nm)
        )
        return n_pe / n_photons

    @u.quantity_input
    def get_scaled_nsb_rate(self, nsb_flux: NSB_FLUX_UNIT):
        scale = nsb_flux / self._nsb_flux_300_650
        return self._nsb_rate_inside_pixel * scale

    @property
    def nominal_nsb_rate(self):
        return self.get_scaled_nsb_rate(u.Quantity(0.24, NSB_FLUX_UNIT))

    @property
    def nominal_nsb_photon_rate(self):
        return self.nominal_nsb_rate / self._nsb_rate_inside_pixel * self._nsb_photon_rate_at_pixel

    def get_nsb_photon_rate_for_led(self, nsb_pe_rate, led_wavelength):
        pde_at_wavelength = self.pde[self.wavelength == led_wavelength]
        nsb_photon_rate = nsb_pe_rate / pde_at_wavelength
        return nsb_photon_rate

    def get_led_nsb_ratio(self, led_wavelength):
        """Ratio between photons observed from NSB spectrum, and photons observed from LED"""
        pde_at_wavelength = self.pde[self.wavelength == led_wavelength]
        return self._nsb_flux_at_pixel / self._nsb_flux_inside_pixel * pde_at_wavelength

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
    def maximum_nsb_rate(self):
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
        return ((ref_wavelength / self.wavelength)**2 * self._atmospheric_transmissivity
                * self._cherenkov_scale)

    @property
    def _cherenkov_diff_flux_at_mirror(self):
        return self._cherenkov_diff_flux_on_ground * self.telescope_transmissivity

    @property
    def _cherenkov_diff_flux_at_camera(self):
        return self._cherenkov_diff_flux_at_mirror * self.mirror_reflectivity

    @property
    def _cherenkov_diff_flux_at_pixel(self):
        return self._cherenkov_diff_flux_at_camera * self.window_transmissivity

    @property
    def _cherenkov_flux_at_pixel(self):
        wl_min = u.Quantity(200, u.nm)
        wl_max = u.Quantity(999, u.nm)
        return self._integrate_cherenkov(self._cherenkov_diff_flux_at_pixel, wl_min, wl_max)

    @property
    def _cherenkov_diff_flux_inside_pixel(self):
        return self._cherenkov_diff_flux_at_pixel * self.pde

    @property
    def _cherenkov_flux_inside_pixel(self):
        wl_min = u.Quantity(200, u.nm)
        wl_max = u.Quantity(999, u.nm)
        return self._integrate_cherenkov(self._cherenkov_diff_flux_inside_pixel, wl_min, wl_max)

    @property
    def n_cherenkov_photoelectrons(self):
        return self._integrate_cherenkov(
            self._cherenkov_diff_flux_inside_pixel,
            u.Quantity(200, u.nm),
            u.Quantity(999, u.nm)
        ) * self.pixel_fill_factor

    def get_led_cherenkov_ratio(self, led_wavelength):
        """Ratio between photons observed from Cherenkov spectrum, and photons observed from LED"""
        pde_at_wavelength = self.pde[self.wavelength == led_wavelength]
        return self._cherenkov_flux_at_pixel / self._cherenkov_flux_inside_pixel * pde_at_wavelength

    @property
    def camera_cherenkov_pde(self):
        n_pe = self.n_cherenkov_photoelectrons

        # According to Konrad's spreadsheet:
        # normalizing to photons outside sensible range makes no sense
        n_photons = self._integrate_cherenkov(
            self._cherenkov_diff_flux_at_camera,
            u.Quantity(300, u.nm),
            u.Quantity(550, u.nm)
        )
        return n_pe / n_photons

    @property
    def telescope_cherenkov_pde(self):
        n_pe = self.n_cherenkov_photoelectrons

        # According to Konrad's spreadsheet:
        # normalizing to photons outside sensible range makes no sense
        n_photons = self._integrate_cherenkov(
            self._cherenkov_diff_flux_at_mirror,
            u.Quantity(300, u.nm),
            u.Quantity(550, u.nm)
        )
        return n_pe / n_photons

    @property
    def _cherenkov_flux_300_550(self):
        """Sum(C1) from 300 - 550 nm (Konrad's spreadsheet)"""
        return self._integrate_cherenkov(
            self._cherenkov_diff_flux_on_ground, u.Quantity(300, u.nm), u.Quantity(550, u.nm)
        )

    @property
    def _cherenkov_diff_flux_inside_pixel_bypass_telescope(self):
        return self._cherenkov_diff_flux_on_ground * self.window_transmissivity * self.pde

    @property
    def _cherenkov_flux_300_550_inside_pixel_bypass_telescope(self):
        return self._integrate_cherenkov(
            self._cherenkov_diff_flux_inside_pixel_bypass_telescope,
            u.Quantity(300, u.nm),
            u.Quantity(550, u.nm)
        )

    @property
    def B_TEL_1170_pde(self):
        """As defined by Konrad ("broken" definition)"""
        return (self._cherenkov_flux_300_550_inside_pixel_bypass_telescope /
                self._cherenkov_flux_300_550 *
                self.pixel_fill_factor)

    @property
    def camera_signal_to_noise(self):
        return self.camera_cherenkov_pde / np.sqrt(self.camera_nsb_pde)

    @property
    def telescope_signal_to_noise(self):
        """B-TEL-0090 Signal to Noise"""
        return self.telescope_cherenkov_pde / np.sqrt(self.telescope_nsb_pde)

    @classmethod
    def from_prod4(cls):
        # Read telescope arrays
        df_tel = pd.read_csv(PROD4_PATH_TEL)
        telescope_transmissivity = df_tel['telescope_transmissivity'].values
        mirror_reflectivity = df_tel['mirror_reflectivity'].values

        # Read camera window transmissivity
        df_window = pd.read_csv(PROD4_PATH_WINDOW)
        window_transmissivity = df_window['window_transmissivity'].values

        # Read camera pde
        df_pde = pd.read_csv(PROD4_PATH_PDE)
        pde = df_pde['pde'].values

        # Read scalar quantities
        with open(PROD4_PATH_QUAN, 'r') as stream:
            quantities = yaml.safe_load(stream)

        pixel_diameter = u.Quantity(quantities["pixel_diameter"], 'm')
        pixel_fill_factor = quantities["pixel_fill_factor"]
        focal_length = u.Quantity(quantities["focal_length"], 'm/radian')
        mirror_area = u.Quantity(quantities["mirror_area"], 'm2')

        return cls(
            pixel_diameter=pixel_diameter,
            pixel_fill_factor=pixel_fill_factor,
            focal_length=focal_length,
            mirror_area=mirror_area,
            telescope_transmissivity=telescope_transmissivity,
            mirror_reflectivity=mirror_reflectivity,
            window_transmissivity=window_transmissivity,
            pde=pde,
        )

    @classmethod
    def from_sstcam_lvr3_uncoated(cls, incidence_angle=0):
        """0 incidence angle"""

        # Read telescope arrays TODO: update
        df_tel = pd.read_csv(PROD4_PATH_TEL)
        telescope_transmissivity = df_tel['telescope_transmissivity'].values
        mirror_reflectivity = df_tel['mirror_reflectivity'].values

        # Read camera window transmissivity
        window = WindowDurhamNeedle()
        window_transmissivity = window.interpolate(incidence_angle)

        # Read camera pde
        pde_vs_wavelength = PDEvsWavelength.LVR3_75um_6mm()
        pde = pde_vs_wavelength.interpolate(incidence_angle)

        # Read scalar quantities
        with open(SSTCAM_PATH_QUAN, 'r') as stream:
            quantities = yaml.safe_load(stream)

        pixel_diameter = u.Quantity(quantities["pixel_diameter"], 'm')
        pixel_fill_factor = quantities["pixel_fill_factor"]
        focal_length = u.Quantity(quantities["focal_length"], 'm/radian')
        mirror_area = u.Quantity(quantities["mirror_area"], 'm2')

        return cls(
            pixel_diameter=pixel_diameter,
            pixel_fill_factor=pixel_fill_factor,
            focal_length=focal_length,
            mirror_area=mirror_area,
            telescope_transmissivity=telescope_transmissivity,
            mirror_reflectivity=mirror_reflectivity,
            window_transmissivity=window_transmissivity,
            pde=pde,
        )

    @classmethod
    def from_sstcam_lct5_resin(cls):
        """0 incidence angle"""

        # Read telescope arrays TODO: update
        df_tel = pd.read_csv(PROD4_PATH_TEL)
        telescope_transmissivity = df_tel['telescope_transmissivity'].values
        mirror_reflectivity = df_tel['mirror_reflectivity'].values

        # Read camera window transmissivity
        window = WindowDurhamNeedle()
        window_transmissivity = window.df['0_measured'].values

        # Read camera pde
        pde = read_lct5_resin_coated()

        # Read scalar quantities
        with open(SSTCAM_PATH_QUAN, 'r') as stream:
            quantities = yaml.safe_load(stream)

        pixel_diameter = u.Quantity(quantities["pixel_diameter"], 'm')
        pixel_fill_factor = quantities["pixel_fill_factor"]
        focal_length = u.Quantity(quantities["focal_length"], 'm/radian')
        mirror_area = u.Quantity(quantities["mirror_area"], 'm2')

        return cls(
            pixel_diameter=pixel_diameter,
            pixel_fill_factor=pixel_fill_factor,
            focal_length=focal_length,
            mirror_area=mirror_area,
            telescope_transmissivity=telescope_transmissivity,
            mirror_reflectivity=mirror_reflectivity,
            window_transmissivity=window_transmissivity,
            pde=pde,
        )

    @classmethod
    def from_sstcam_prototype_sipm(cls):
        """0 incidence angle"""

        # Read telescope arrays TODO: update
        df_tel = pd.read_csv(PROD4_PATH_TEL)
        telescope_transmissivity = df_tel['telescope_transmissivity'].values
        mirror_reflectivity = df_tel['mirror_reflectivity'].values

        # Read camera window transmissivity
        window = WindowDurhamNeedle()
        window_transmissivity = window.df['0_measured'].values

        # Read camera pde
        pde = read_prototype()

        # Read scalar quantities
        with open(SSTCAM_PATH_QUAN, 'r') as stream:
            quantities = yaml.safe_load(stream)

        pixel_diameter = u.Quantity(quantities["pixel_diameter"], 'm')
        pixel_fill_factor = quantities["pixel_fill_factor"]
        focal_length = u.Quantity(quantities["focal_length"], 'm/radian')
        mirror_area = u.Quantity(quantities["mirror_area"], 'm2')

        return cls(
            pixel_diameter=pixel_diameter,
            pixel_fill_factor=pixel_fill_factor,
            focal_length=focal_length,
            mirror_area=mirror_area,
            telescope_transmissivity=telescope_transmissivity,
            mirror_reflectivity=mirror_reflectivity,
            window_transmissivity=window_transmissivity,
            pde=pde,
        )

    @classmethod
    def from_sstcam(cls, fov_angle=0, pde_at_450nm=None, window=None):
        if window is None:
            window = WindowDurhamNeedle()

        # Read telescope arrays TODO: updated & vs axis angle
        df_tel = pd.read_csv(PROD4_PATH_TEL)
        telescope_transmissivity = df_tel['telescope_transmissivity'].values
        mirror_reflectivity = df_tel['mirror_reflectivity'].values

        # Weight window transmission by angular distribution of photon incidence
        weighted_window_transmissivity = window.weight_by_incidence_angle(
            off_axis_angle=fov_angle
        )

        # Read camera pde
        pde_vs_wavelength = PDEvsWavelength.LVR3_75um_6mm()
        if pde_at_450nm:
            pde_vs_wavelength.scale(u.Quantity(450, u.nm), pde_at_450nm)
        weighted_pde = pde_vs_wavelength.weight_by_incidence_angle(
            off_axis_angle=fov_angle
        )

        # Read scalar quantities
        with open(SSTCAM_PATH_QUAN, 'r') as stream:
            quantities = yaml.safe_load(stream)

        pixel_diameter = u.Quantity(quantities["pixel_diameter"], 'm')
        pixel_fill_factor = quantities["pixel_fill_factor"]
        focal_length = u.Quantity(quantities["focal_length"], 'm/radian')
        # TODO: vs axis angle? Or already accounted for in incidence angle datasets?
        mirror_area = u.Quantity(quantities["mirror_area"], 'm2')

        return cls(
            pixel_diameter=pixel_diameter,
            pixel_fill_factor=pixel_fill_factor,
            focal_length=focal_length,
            mirror_area=mirror_area,
            telescope_transmissivity=telescope_transmissivity,
            mirror_reflectivity=mirror_reflectivity,
            window_transmissivity=weighted_window_transmissivity,
            pde=weighted_pde,
        )


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    lvr3 = CameraEfficiency.from_sstcam_lvr3_uncoated()
    lvr3.scale_pde(u.Quantity(470, u.nm), 0.532)

    lct5 = CameraEfficiency.from_sstcam_lct5_resin()
    lct5.scale_pde(u.Quantity(470, u.nm), 0.532)

    proto = CameraEfficiency.from_sstcam_prototype_sipm()
    #
    print("LVR3: ", lvr3.camera_cherenkov_pde, lvr3.B_TEL_1170_pde, lvr3.telescope_signal_to_noise)
    print("LCT5: ", lct5.camera_cherenkov_pde, lct5.B_TEL_1170_pde, lct5.telescope_signal_to_noise)
    print("LVR3: ", proto.camera_cherenkov_pde, proto.B_TEL_1170_pde, proto.telescope_signal_to_noise)

    plt.plot(lvr3.wavelength, lvr3.pde, label=f"LVR3 (ChPDE: {lvr3.camera_cherenkov_pde:.2f}, S/N: {lvr3.telescope_signal_to_noise:.2f})")
    plt.plot(lct5.wavelength, lct5.pde, label=f"LCT5 Resin (ChPDE: {lct5.camera_cherenkov_pde:.2f}, S/N: {lct5.telescope_signal_to_noise:.2f})")
    plt.plot(proto.wavelength, proto.pde, label=f"Prototype (ChPDE: {proto.camera_cherenkov_pde:.2f}, S/N: {proto.telescope_signal_to_noise:.2f})")
    plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("PDE")

    plt.show()
