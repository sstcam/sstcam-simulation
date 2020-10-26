from sstcam_simulation.event.photoelectrons import Photoelectrons
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation.event.acquisition import EventAcquisition
from sstcam_simulation.camera import Camera, SSTCameraMapping
import numpy as np
import pytest


@pytest.fixture(scope="module")
def acquisition():
    camera = Camera(continuous_readout_duration=1000, mapping=SSTCameraMapping(n_pixels=2))
    acquisition = EventAcquisition(camera=camera)
    return acquisition


def test_get_continuous_readout(acquisition):
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1]), time=np.array([30, 40]), charge=np.array([1.0, 2.0])
    )
    readout = acquisition.get_continuous_readout(photoelectrons)
    integral = readout.sum(1) * acquisition.camera.continuous_readout_sample_width
    argmax = readout.argmax(1) * acquisition.camera.continuous_readout_sample_width
    np.testing.assert_allclose(integral, photoelectrons.charge)
    np.testing.assert_allclose(argmax, photoelectrons.time)

    # Outside of readout duration
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1, 1]),
        time=np.array([999, 1000, 1001]),
        charge=np.array([1.0, 2.0, 1.0])
    )
    readout = acquisition.get_continuous_readout(photoelectrons)
    integral = readout.sum() * acquisition.camera.continuous_readout_sample_width
    argmax = readout.argmax() * acquisition.camera.continuous_readout_sample_width
    assert integral < 1
    np.testing.assert_allclose(argmax, 999)


def test_get_continuous_readout_with_noise():
    pulse = GaussianPulse()
    noise = GaussianNoise(stddev=pulse.height, seed=1)
    camera = Camera(
        continuous_readout_duration=1000,
        n_waveform_samples=1000,
        photoelectron_pulse=pulse,
        readout_noise=noise,
        mapping=SSTCameraMapping(n_pixels=1)
    )
    acquisition = EventAcquisition(camera=camera, seed=1)
    photoelectrons = Photoelectrons(
        pixel=np.array([], dtype=np.int), time=np.array([]), charge=np.array([])
    )
    readout = acquisition.get_continuous_readout(photoelectrons)
    stddev_pe = readout.std() / camera.photoelectron_pulse.height
    np.testing.assert_allclose(stddev_pe, 1, rtol=1e-2)

    waveform = acquisition.get_sampled_waveform(readout)
    predicted_stddev = noise.stddev / np.sqrt(camera.continuous_readout_sample_division)
    np.testing.assert_allclose(waveform.std(), predicted_stddev, rtol=1e-2)


def test_get_sampled_waveform():
    camera = Camera()
    acquisition = EventAcquisition(camera=camera)
    n_pixels = camera.mapping.n_pixels
    time_axis = camera.continuous_readout_time_axis
    n_continuous_samples = time_axis.size
    n_samples = camera.n_waveform_samples
    sample = camera.get_waveform_sample_from_time
    csample = camera.get_continuous_readout_sample_from_time
    cwidth = camera.continuous_readout_sample_width
    continuous_readout = np.zeros((n_pixels, n_continuous_samples))
    continuous_readout[0, csample(30.0):csample(30.5)] = 100
    continuous_readout[2, csample(40.0):csample(41.0)] = 100

    waveform = acquisition.get_sampled_waveform(continuous_readout)
    assert waveform.shape == (n_pixels, n_samples)
    assert waveform[0].sum() == continuous_readout[0].sum() * cwidth
    assert waveform[2].sum() == continuous_readout[2].sum() * cwidth
    assert waveform[0].argmax() == sample(30.0)
    assert waveform[2].argmax() == sample(40.0)

    waveform = acquisition.get_sampled_waveform(continuous_readout, 30)
    assert waveform.shape == (n_pixels, n_samples)
    assert waveform[0].sum() == continuous_readout[0].sum() * cwidth
    assert waveform[2].sum() == continuous_readout[2].sum() * cwidth
    assert waveform[0].argmax() == sample(20.0)
    assert waveform[2].argmax() == sample(30.0)

    waveform = acquisition.get_sampled_waveform(continuous_readout, 25)
    assert waveform.shape == (n_pixels, n_samples)
    assert waveform[0].sum() == continuous_readout[0].sum() * cwidth
    assert waveform[2].sum() == continuous_readout[2].sum() * cwidth
    assert waveform[0].argmax() == sample(25.0)
    assert waveform[2].argmax() == sample(35.0)

    # Out of bounds
    with pytest.raises(ValueError):
        acquisition.get_sampled_waveform(continuous_readout, 10)
    with pytest.raises(ValueError):
        acquisition.get_sampled_waveform(continuous_readout, 900)

    # Single Pixel
    camera = Camera(mapping=SSTCameraMapping(n_pixels=1))
    acquisition = EventAcquisition(camera=camera)
    n_pixels = camera.mapping.n_pixels
    time_axis = camera.continuous_readout_time_axis
    n_continuous_samples = time_axis.size
    n_samples = camera.waveform_duration * camera.waveform_sample_width
    continuous_readout = np.zeros((n_pixels, n_continuous_samples))
    continuous_readout[0, csample(30.0):csample(30.5)] = 100

    waveform = acquisition.get_sampled_waveform(continuous_readout)
    assert waveform.shape == (n_pixels, n_samples)
    assert waveform[0].sum() == continuous_readout[0].sum() * cwidth
    assert waveform[0].argmax() == sample(30.0)

    waveform = acquisition.get_sampled_waveform(continuous_readout, 30)
    assert waveform.shape == (n_pixels, n_samples)
    assert waveform[0].sum() == continuous_readout[0].sum() * cwidth
    assert waveform[0].argmax() == sample(20.0)


def test_get_sampled_waveform_sample_width():
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),
    )
    pe = Photoelectrons(pixel=np.array([0]), time=np.array([40]), charge=np.array([1]))
    acquisition = EventAcquisition(camera=camera, seed=1)
    readout = acquisition.get_continuous_readout(pe)
    waveform = acquisition.get_sampled_waveform(readout)
    print(waveform.max())
