from sstcam_simulation.camera import Camera
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation.camera.spe import SiPMGentileSPE


def test_camera():
    # Default Initialisation
    Camera()

    # Custom Initialisation
    reference_pulse = GaussianPulse()
    spectrum = SiPMGentileSPE()
    camera = Camera(
        n_pixels=2048,
        reference_pulse=reference_pulse,
        single_photoelectron_spectrum=spectrum
    )
    assert camera.n_pixels == 2048
    assert camera.reference_pulse == reference_pulse
    assert camera.single_photoelectron_spectrum == spectrum
