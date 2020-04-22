from sstcam_simulation.event.photoelectrons import Photoelectrons
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation.event.acquisition import (
    EventAcquisition,
    sum_superpixels,
    add_coincidence_window,
)
from sstcam_simulation.camera import Camera, SSTCameraMapping
import numpy as np
import pytest


@pytest.fixture(scope="module")
def acquisition():
    camera = Camera(mapping=SSTCameraMapping(n_pixels=2))
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


def test_get_continuous_readout_with_noise():
    pulse = GaussianPulse()
    noise = GaussianNoise(stddev=pulse.peak_height, seed=1)
    camera = Camera(
        reference_pulse=pulse,
        electronic_noise=noise,
        mapping=SSTCameraMapping(n_pixels=1)
    )
    acquisition = EventAcquisition(camera=camera, seed=1)
    photoelectrons = Photoelectrons(
        pixel=np.array([], dtype=np.int), time=np.array([]), charge=np.array([])
    )
    readout = acquisition.get_continuous_readout(photoelectrons)
    stddev_pe = readout.std() / camera.reference_pulse.peak_height
    np.testing.assert_allclose(stddev_pe, 1, rtol=1e-2)


def test_sum_superpixels():
    n_superpixels = 2
    n_samples = 10
    continuous_readout = np.ones((5, n_samples)) * np.arange(1, 6)[:, None]
    superpixel = np.array([0, 0, 1, 0, 1])
    superpixel_sum = sum_superpixels(continuous_readout, superpixel, n_superpixels)
    assert superpixel_sum.shape == (n_superpixels, n_samples)
    assert np.array_equal(superpixel_sum[0], np.full(10, 7))
    assert np.array_equal(superpixel_sum[1], np.full(10, 8))


def test_add_coincidence_window(acquisition):
    n_pixels = acquisition.camera.mapping.n_pixels
    time_axis = acquisition.camera.continuous_readout_time_axis
    n_samples = time_axis.size
    division = acquisition.camera.continuous_readout_sample_division
    window = acquisition.camera.coincidence_window * division
    above_threshold = np.zeros((n_pixels, n_samples), dtype=np.bool)
    above_threshold[0, 100:101] = True
    above_threshold[1, 200:500] = True
    above_threshold[1, 510:520] = True
    trigger_readout_expected = np.zeros((n_pixels, n_samples), dtype=np.bool)
    trigger_readout_expected[0, 100:101+window] = True
    trigger_readout_expected[1, 200:500+window] = True
    trigger_readout_expected[1, 510:520+window] = True
    trigger_readout = add_coincidence_window(above_threshold, window)
    assert np.array_equal(trigger_readout, trigger_readout_expected)


def test_update_trigger_threshold():
    camera = Camera(trigger_threshold=5)
    acquisition = EventAcquisition(camera=camera)
    assert acquisition.camera.trigger_threshold == 5
    camera.update_trigger_threshold(6)
    assert acquisition.camera.trigger_threshold == 6


def test_get_digital_trigger_readout(acquisition):
    n_pixels = acquisition.camera.mapping.n_pixels
    n_superpixels = acquisition.camera.mapping.n_superpixels
    time_axis = acquisition.camera.continuous_readout_time_axis
    n_samples = time_axis.size
    division = acquisition.camera.continuous_readout_sample_division
    window = acquisition.camera.coincidence_window * division
    continuous_readout = np.zeros((n_pixels, n_samples))
    continuous_readout[0, 100:101] = 100
    continuous_readout[0, 200:500] = 0.1
    continuous_readout[1, 200:500] = 100
    continuous_readout[1, 510:520] = 100
    trigger_readout_expected = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_readout_expected[0, 100:101+window] = True
    trigger_readout_expected[0, 200:500+window] = True
    trigger_readout_expected[0, 510:520+window] = True
    trigger_readout = acquisition.get_superpixel_digital_trigger_line(continuous_readout)
    assert np.array_equal(trigger_readout, trigger_readout_expected)


def test_get_n_superpixel_triggers(acquisition):
    n_superpixels = 2
    n_samples = acquisition.camera.continuous_readout_time_axis.size
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_readout[0, 10] = True
    trigger_readout[0, 15] = True
    trigger_readout[1, 100] = True
    trigger_readout[1, 150] = True
    trigger_readout[1, 200] = True
    n_triggers = acquisition.get_n_superpixel_triggers(trigger_readout)
    assert np.array_equal(n_triggers, np.array([2, 3], dtype=np.int))


def test_get_backplane_trigger():
    camera = Camera()
    acquisition = EventAcquisition(camera=camera)
    n_superpixels = camera.mapping.n_superpixels
    n_samples = camera.continuous_readout_time_axis.size
    csample = camera.get_continuous_readout_sample_from_time
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_time, trigger_pair = acquisition.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)

    # No overlap (after sampling)
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_readout[0, csample(1.00):csample(2.00)] = True
    trigger_readout[0, csample(10.0):csample(12.0)] = True
    trigger_readout[1, csample(11.5):csample(13.0)] = True
    trigger_readout[2, csample(20.0):csample(22.0)] = True
    trigger_readout[3, csample(21.5):csample(23.0)] = True
    trigger_time, trigger_pair = acquisition.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)

    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_readout[0, csample(1.00):csample(2.00)] = True
    trigger_readout[0, csample(10.0):csample(12.5)] = True
    trigger_readout[1, csample(11.5):csample(13.0)] = True
    trigger_readout[2, csample(20.0):csample(22.5)] = True
    trigger_readout[3, csample(21.5):csample(23.0)] = True
    trigger_time, trigger_pair = acquisition.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (2,)
    assert trigger_pair.shape == (2, 2)
    assert np.array_equal(trigger_time, np.array([12, 22]))
    assert np.array_equal(trigger_pair, np.array([[0, 1], [2, 3]]))

    # Single pixel
    camera = Camera(mapping=SSTCameraMapping(n_pixels=1))
    acquisition = EventAcquisition(camera=camera)
    n_superpixels = camera.mapping.n_superpixels
    n_samples = camera.continuous_readout_time_axis.size
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_readout[0, 10:20] = True
    trigger_readout[0, 100:125] = True
    trigger_time, trigger_pair = acquisition.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)


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
