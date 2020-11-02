from sstcam_simulation.camera.coupling import Coupling, ACFilterCoupling, ACOffsetCoupling
from sstcam_simulation.camera.spe import SiPMGentileSPE, SiPMPrompt
from sstcam_simulation import Camera, PhotoelectronSource, EventAcquisition, \
    SSTCameraMapping
from sstcam_simulation.camera.pulse import GaussianPulse
import pytest
import numpy as np


@pytest.mark.parametrize("coupling_class", Coupling.__subclasses__())
def test_coupling(coupling_class):
    coupling = coupling_class()
    readout = np.ones((2048, 128))
    coupled = coupling.apply_to_readout(readout)
    assert coupled.shape == readout.shape


def get_camera_and_readout(pulse_width, mv_per_pe, spectrum, nsb_rate):
    camera = Camera(
        continuous_readout_duration=int(2e5),
        n_waveform_samples=int(2e5),
        mapping=SSTCameraMapping(n_pixels=1),
        photoelectron_pulse=GaussianPulse(sigma=pulse_width, mv_per_pe=mv_per_pe),
        photoelectron_spectrum=spectrum
    )
    source = PhotoelectronSource(camera=camera, seed=1)
    acquisition = EventAcquisition(camera=camera, seed=1)
    pe = source.get_nsb(nsb_rate)
    return camera, acquisition.get_continuous_readout(pe)


def test_ac_filter_coupling():
    camera, readout = get_camera_and_readout(
        pulse_width=10,
        mv_per_pe=None,
        nsb_rate=100,
        spectrum=SiPMGentileSPE(opct=0.2)
    )
    coupling = ACFilterCoupling(order=1, critical_frequency=160e3)
    coupled = coupling.apply_to_readout(readout)
    np.testing.assert_allclose(coupled.mean(), 0, atol=0.5)

    camera, readout = get_camera_and_readout(
        pulse_width=10,
        mv_per_pe=3,
        nsb_rate=1000,
        spectrum=SiPMGentileSPE(opct=0.2)
    )
    coupling = ACFilterCoupling(order=1, critical_frequency=160e3)
    coupled = coupling.apply_to_readout(readout)
    np.testing.assert_allclose(coupled.mean(), 0, atol=0.5)


def test_ac_offset_coupling():
    nsb_rate = 100
    camera, readout = get_camera_and_readout(
        pulse_width=10,
        mv_per_pe=None,
        nsb_rate=nsb_rate,
        spectrum=SiPMGentileSPE(opct=0.2, normalise_charge=True)
    )
    pulse_area = camera.photoelectron_pulse.area
    spectrum_average = camera.photoelectron_spectrum.average
    coupling = ACOffsetCoupling(nsb_rate, pulse_area, spectrum_average)
    coupled = coupling.apply_to_readout(readout)
    np.testing.assert_allclose(coupled.mean(), 0, atol=0.5)

    nsb_rate = 10000
    camera, readout = get_camera_and_readout(
        pulse_width=10,
        mv_per_pe=3,
        nsb_rate=nsb_rate,
        spectrum=SiPMGentileSPE(opct=0.2, normalise_charge=True)
    )
    pulse_area = camera.photoelectron_pulse.area
    spectrum_average = camera.photoelectron_spectrum.average
    coupling = ACOffsetCoupling(nsb_rate, pulse_area, spectrum_average)
    coupled = coupling.apply_to_readout(readout)
    np.testing.assert_allclose(coupled.mean(), 0, atol=0.5)

    nsb_rate = 10000
    camera, readout = get_camera_and_readout(
        pulse_width=10,
        mv_per_pe=3,
        nsb_rate=nsb_rate,
        spectrum=SiPMGentileSPE(opct=0.4, normalise_charge=False)
    )
    pulse_area = camera.photoelectron_pulse.area
    spectrum_average = camera.photoelectron_spectrum.average
    coupling = ACOffsetCoupling(nsb_rate, pulse_area, spectrum_average)
    coupled = coupling.apply_to_readout(readout)
    np.testing.assert_allclose(coupled.mean(), 0, atol=0.5)

    nsb_rate = 10000
    camera, readout = get_camera_and_readout(
        pulse_width=10,
        mv_per_pe=3,
        nsb_rate=nsb_rate,
        spectrum=SiPMPrompt(opct=0.4, normalise_charge=True)
    )
    pulse_area = camera.photoelectron_pulse.area
    spectrum_average = camera.photoelectron_spectrum.average
    coupling = ACOffsetCoupling(nsb_rate, pulse_area, spectrum_average)
    coupled = coupling.apply_to_readout(readout)
    np.testing.assert_allclose(coupled.mean(), 0, atol=0.5)

    nsb_rate = 10000
    camera, readout = get_camera_and_readout(
        pulse_width=10,
        mv_per_pe=3,
        nsb_rate=nsb_rate,
        spectrum=SiPMPrompt(opct=0.4, normalise_charge=False)
    )
    pulse_area = camera.photoelectron_pulse.area
    spectrum_average = camera.photoelectron_spectrum.average
    coupling = ACOffsetCoupling(nsb_rate, pulse_area, spectrum_average)
    coupled = coupling.apply_to_readout(readout)
    np.testing.assert_allclose(coupled.mean(), 0, atol=0.5)
