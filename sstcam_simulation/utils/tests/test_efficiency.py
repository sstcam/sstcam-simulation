from sstcam_simulation.utils.efficiency import CameraEfficiency
import numpy as np
from astropy import units as u
import pytest


@pytest.fixture(scope="module")
def prod4_cam_eff():
    return CameraEfficiency.from_prod4()


def test_prod4_pixel_active_solid_angle(prod4_cam_eff):
    pixel_active_solid_angle = prod4_cam_eff._pixel_active_solid_angle.to_value("μsr")
    np.testing.assert_allclose(pixel_active_solid_angle, 8.3, rtol=1e-3)


def test_prod4_pixel_fill_factor(prod4_cam_eff):
    fill_factor = prod4_cam_eff.pixel_fill_factor
    np.testing.assert_allclose(fill_factor, 0.939, rtol=1e-3)


def test_scale_pde(prod4_cam_eff):
    original_pde = prod4_cam_eff.pde.copy()

    prod4_cam_eff.scale_pde(u.Quantity(410, u.nm), 0.5)
    pde_at_wavelength = prod4_cam_eff.pde[prod4_cam_eff.wavelength == 410]
    np.testing.assert_allclose(pde_at_wavelength, 0.5, rtol=1e-2)
    prod4_cam_eff.scale_pde(u.Quantity(410, u.nm), 0.25)
    pde_at_wavelength = prod4_cam_eff.pde[prod4_cam_eff.wavelength == 410]
    np.testing.assert_allclose(pde_at_wavelength, 0.25, rtol=1e-2)

    prod4_cam_eff.reset_pde_scale()
    np.testing.assert_allclose(prod4_cam_eff.pde, original_pde, rtol=1e-2)


def test_prod4_nsb_rate_inside_pixel(prod4_cam_eff):
    nsb_rate = prod4_cam_eff._nsb_rate_inside_pixel.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.044, rtol=1e-2)


def test_prod4_integrate_nsb(prod4_cam_eff):
    nsb_diff_flux = prod4_cam_eff._nsb_diff_flux_on_ground
    nsb_flux = prod4_cam_eff._integrate_nsb(nsb_diff_flux, 300 * u.nm, 550 * u.nm)
    nsb_flux = nsb_flux.to_value("1/(cm² ns sr)")
    np.testing.assert_allclose(nsb_flux, 0.116, rtol=1e-2)
    nsb_flux = prod4_cam_eff._integrate_nsb(nsb_diff_flux, 300 * u.nm, 650 * u.nm)
    nsb_flux = nsb_flux.to_value("1/(cm² ns sr)")
    np.testing.assert_allclose(nsb_flux, 0.263, rtol=1e-2)


def test_prod4_nominal_nsb(prod4_cam_eff):
    nsb_rate = prod4_cam_eff.nominal_nsb_rate.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.04, rtol=1e-2)


def test_prod4_high_nsb(prod4_cam_eff):
    nsb_rate = prod4_cam_eff.high_nsb_rate.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.72, rtol=1e-2)


def test_prod4_nominal_moonlight(prod4_cam_eff):
    nsb_rate = prod4_cam_eff.nominal_moonlight_rate.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.145, rtol=1e-2)


def test_prod4_high_moonlight(prod4_cam_eff):
    nsb_rate = prod4_cam_eff.high_moonlight_rate.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.75, rtol=1e-2)


def test_prod4_integrate_cherenkov(prod4_cam_eff):
    diff_flux = prod4_cam_eff._cherenkov_diff_flux_inside_pixel
    flux = prod4_cam_eff._integrate_cherenkov(diff_flux, 300 * u.nm, 550 * u.nm)
    np.testing.assert_allclose(flux, 50.12, rtol=1e-2)
    flux = prod4_cam_eff._integrate_cherenkov(diff_flux, 200 * u.nm, 999 * u.nm)
    np.testing.assert_allclose(flux, 56.66, rtol=1e-2)


def test_prod4_effective_cherenkov_pde(prod4_cam_eff):
    np.testing.assert_allclose(prod4_cam_eff.effective_cherenkov_pde, 0.441, rtol=1e-3)
    prod4_cam_eff.scale_pde(u.Quantity(410, u.nm), 0.5)
    np.testing.assert_allclose(prod4_cam_eff.effective_cherenkov_pde, 0.451, rtol=1e-3)
    prod4_cam_eff.scale_pde(u.Quantity(410, u.nm), 0.25)
    np.testing.assert_allclose(prod4_cam_eff.effective_cherenkov_pde, 0.2207, rtol=1e-3)
    prod4_cam_eff.reset_pde_scale()


def test_prod4_B_TEL_1170_pde(prod4_cam_eff):
    np.testing.assert_allclose(prod4_cam_eff._cherenkov_flux_300_550, 168.41, rtol=1e-4)
    f = prod4_cam_eff._cherenkov_flux_300_550_inside_pixel_bypass_telescope
    np.testing.assert_allclose(f, 69.58, rtol=1e-4)
    np.testing.assert_allclose(prod4_cam_eff.B_TEL_1170_pde, 0.388, rtol=1e-4)
