from sstcam_simulation.camera import Camera, PixelMapping
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
        reference_pulse=reference_pulse,
        photoelectron_spectrum=spectrum
    )
    assert camera.pixel.n_pixels == 2048
    assert camera.reference_pulse == reference_pulse
    assert camera.photoelectron_spectrum == spectrum


def test_continuous_readout():
    camera = Camera()
    pixel = np.array([0], dtype=np.int)
    time = np.array([camera.continuous_readout_length // 2])
    charge = np.array([2])


def test_n_pixels():
    camera = Camera()
    assert camera.pixel.n_pixels == 2048
    assert camera.superpixel.n_superpixels == 512

    camera = Camera(pixel=PixelMapping(n_pixels=2))
    assert camera.pixel.n_pixels == 2
    assert camera.superpixel.n_superpixels == 1
