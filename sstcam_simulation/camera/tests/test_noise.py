from sstcam_simulation.camera.noise import ElectronicNoise
import pytest
import numpy as np


@pytest.mark.parametrize("noise_class", ElectronicNoise.__subclasses__())
def test_noise(noise_class):
    noise = noise_class()
    readout = np.ones((2048, 128))
    noisy = noise.add_to_readout(readout)
    assert noisy.shape == readout.shape
