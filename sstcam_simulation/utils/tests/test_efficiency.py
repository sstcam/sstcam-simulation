from sstcam_simulation.utils.efficiency import CameraEfficiency
import numpy as np
from astropy import units as u


def test_prod4_pixel_active_solid_angle():
    cam_eff = CameraEfficiency()
    pixel_active_solid_angle = cam_eff._pixel_active_solid_angle.to_value("μsr")
    np.testing.assert_allclose(pixel_active_solid_angle, 8.3, rtol=1e-3)


def test_scale_pde():
    cam_eff = CameraEfficiency()
    cam_eff.scale_pde(u.Quantity(410, u.nm), 0.5)
    pde_at_wavelength = cam_eff.pde[cam_eff.wavelength == 410]
    np.testing.assert_allclose(pde_at_wavelength, 0.5, rtol=1e-2)
    cam_eff.scale_pde(u.Quantity(410, u.nm), 0.25)
    pde_at_wavelength = cam_eff.pde[cam_eff.wavelength == 410]
    np.testing.assert_allclose(pde_at_wavelength, 0.25, rtol=1e-2)


def test_prod4_nsb_rate_inside_pixel():
    cam_eff = CameraEfficiency()
    nsb_rate = cam_eff._nsb_rate_inside_pixel.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.044, rtol=1e-2)


def test_integrate_nsb():
    cam_eff = CameraEfficiency()
    nsb_diff_flux = cam_eff._nsb_diff_flux_on_ground
    nsb_flux = cam_eff._integrate_nsb(nsb_diff_flux, 300 * u.nm, 550 * u.nm)
    nsb_flux = nsb_flux.to_value("1/(cm² ns sr)")
    np.testing.assert_allclose(nsb_flux, 0.116, rtol=1e-2)
    nsb_flux = cam_eff._integrate_nsb(nsb_diff_flux, 300 * u.nm, 650 * u.nm)
    nsb_flux = nsb_flux.to_value("1/(cm² ns sr)")
    np.testing.assert_allclose(nsb_flux, 0.263, rtol=1e-2)


def test_prod4_nominal_nsb():
    cam_eff = CameraEfficiency()
    nsb_rate = cam_eff.nominal_nsb_rate.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.04, rtol=1e-2)


def test_prod4_high_nsb():
    cam_eff = CameraEfficiency()
    nsb_rate = cam_eff.high_nsb_rate.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.72, rtol=1e-2)


def test_prod4_nominal_moonlight():
    cam_eff = CameraEfficiency()
    nsb_rate = cam_eff.nominal_moonlight_rate.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.145, rtol=1e-2)


def test_prod4_high_moonlight():
    cam_eff = CameraEfficiency()
    nsb_rate = cam_eff.high_moonlight_rate.to_value("1/ns")
    np.testing.assert_allclose(nsb_rate, 0.75, rtol=1e-2)


def test_integrate_cherenkov():
    cam_eff = CameraEfficiency()
    diff_flux = cam_eff._cherenkov_diff_flux_inside_pixel
    flux = cam_eff._integrate_cherenkov(diff_flux, 300 * u.nm, 550 * u.nm)
    np.testing.assert_allclose(flux, 50.12, rtol=1e-2)
    flux = cam_eff._integrate_cherenkov(diff_flux, 200 * u.nm, 999 * u.nm)
    np.testing.assert_allclose(flux, 56.66, rtol=1e-2)


def test_effective_cherenkov_pde():
    cam_eff = CameraEfficiency()
    np.testing.assert_allclose(cam_eff.effective_cherenkov_pde, 0.42, rtol=1e-2)
    cam_eff.scale_pde(u.Quantity(410, u.nm), 0.5)
    np.testing.assert_allclose(cam_eff.effective_cherenkov_pde, 0.43, rtol=1e-2)
    cam_eff.scale_pde(u.Quantity(410, u.nm), 0.25)
    np.testing.assert_allclose(cam_eff.effective_cherenkov_pde, 0.213, rtol=1e-2)
