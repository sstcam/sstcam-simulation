from sstcam_simulation.camera import Camera
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation.camera.spe import SiPMGentileSPE
import numpy as np


def test_camera():
    # Default Initialisation
    Camera()

    # Custom Initialisation
    reference_pulse = GaussianPulse()
    spectrum = SiPMGentileSPE()
    camera = Camera(
        n_pixels=2048,
        reference_pulse=reference_pulse,
        photoelectron_spectrum=spectrum
    )
    assert camera.n_pixels == 2048
    assert camera.reference_pulse == reference_pulse
    assert camera.photoelectron_spectrum == spectrum


def test_continuous_readout():
    camera = Camera()
    pixel = np.array([0], dtype=np.int)
    time = np.array([camera.waveform_length//2])
    charge = np.array([2])

    readout = camera.get_continuous_readout(pixel, time, charge)
    integral = readout.sum(1) * camera.continuous_sample_width
    argmax = readout.argmax(1) * camera.continuous_sample_width
    np.testing.assert_allclose(integral, charge)
    np.testing.assert_allclose(argmax, time)
