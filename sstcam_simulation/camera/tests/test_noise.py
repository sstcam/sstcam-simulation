from sstcam_simulation.camera.noise import PerfectElectronics, GaussianNoise, TemplateNoise
import numpy as np


def test_perfect_electronics():
    noise = PerfectElectronics()
    readout = np.ones((2048, 128))
    noisy = noise.add_to_readout(readout)
    assert noisy.shape == readout.shape


def test_gaussian_noise():
    noise = GaussianNoise(stddev=2)
    readout = np.ones((2048, 128))
    noisy = noise.add_to_readout(readout)
    assert noisy.shape == readout.shape
    np.testing.assert_allclose(noisy.std(), 2, rtol=1e-2)


def test_template_noise():
    noise = TemplateNoise(1280, 1e-9, stddev=2)
    readout = np.ones((2048, 1280))
    noisy = noise.add_to_readout(readout)
    assert noisy.shape == readout.shape
    np.testing.assert_allclose(noisy.std(), 2, rtol=1e-2)


def test_template_noise_continuous_readout():
    noise = TemplateNoise(1280*4, 0.25e-9, stddev=2)
    readout = np.ones((2048, 1280*4))
    noisy = noise.add_to_readout(readout)
    assert noisy.shape == readout.shape
    np.testing.assert_allclose(noisy.std(), 2, rtol=1e-2)
