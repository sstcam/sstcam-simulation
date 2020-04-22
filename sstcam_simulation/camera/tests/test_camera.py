from sstcam_simulation.camera import Camera, SSTCameraMapping
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation.camera.spe import SiPMGentileSPE
import numpy as np
import pytest
from dataclasses import FrozenInstanceError


def test_camera():
    # Default Initialisation
    Camera()

    # Custom Initialisation
    reference_pulse = GaussianPulse()
    spectrum = SiPMGentileSPE()
    camera = Camera(reference_pulse=reference_pulse, photoelectron_spectrum=spectrum)
    assert camera.mapping.n_pixels == 2048
    assert camera.reference_pulse == reference_pulse
    assert camera.photoelectron_spectrum == spectrum


def test_n_pixels():
    camera = Camera()
    assert camera.mapping.n_pixels == 2048
    assert camera.mapping.n_superpixels == 512

    camera = Camera(mapping=SSTCameraMapping(n_pixels=2))
    assert camera.mapping.n_pixels == 2
    assert camera.mapping.n_superpixels == 1


def test_read_only():
    camera = Camera(trigger_threshold=5)
    assert camera.trigger_threshold == 5
    with pytest.raises(FrozenInstanceError):
        # noinspection PyDataclass
        camera.trigger_threshold = 6


def test_update_trigger_threshold():
    camera = Camera(trigger_threshold=5)
    assert camera.trigger_threshold == 5
    camera.update_trigger_threshold(6)
    assert camera.trigger_threshold == 6


def test_update_coincidence_window():
    camera = Camera(coincidence_window=5)
    assert camera.coincidence_window == 5
    camera.update_coincidence_window(6)
    assert camera.coincidence_window == 6
