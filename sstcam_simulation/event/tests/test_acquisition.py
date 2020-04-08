from sstcam_simulation.event.photoelectrons import Photoelectrons
from sstcam_simulation.event.acquisition import EventAcquisition, sum_superpixels, \
    add_coincidence_window
from sstcam_simulation.camera import Camera, PixelMapping
import numpy as np
import pytest


@pytest.fixture(scope="module")
def acquisition():
    camera = Camera(pixel=PixelMapping(n_pixels=2))
    acquisition = EventAcquisition(camera=camera)
    return acquisition


def test_get_photoelectrons_per_pixel(acquisition):
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1]), time=np.array([30, 40]), charge=np.array([1.0, 2.0])
    )
    pe = acquisition.get_photoelectrons_per_pixel(photoelectrons)
    assert np.array_equal(pe, np.array([1, 1]))


def test_get_charge_per_pixel(acquisition):
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1]), time=np.array([30, 40]), charge=np.array([1.0, 2.0])
    )
    charge = acquisition.get_charge_per_pixel(photoelectrons)
    assert np.array_equal(charge, np.array([1.0, 2.0]))


def test_get_continuous_readout(acquisition):
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1]), time=np.array([30, 40]), charge=np.array([1.0, 2.0])
    )
    readout = acquisition.get_continuous_readout(photoelectrons)
    integral = readout.sum(1) * acquisition.camera.continuous_sample_width
    argmax = readout.argmax(1) * acquisition.camera.continuous_sample_width
    np.testing.assert_allclose(integral, photoelectrons.charge)
    np.testing.assert_allclose(argmax, photoelectrons.time)


def test_get_continuous_readout_noise():
    camera = Camera(electronic_noise_stddev=1, pixel=PixelMapping(n_pixels=1))
    acquisition = EventAcquisition(camera=camera, seed=1)
    photoelectrons = Photoelectrons(
        pixel=np.array([], dtype=np.int), time=np.array([]), charge=np.array([])
    )
    readout = acquisition.get_continuous_readout(photoelectrons)
    np.testing.assert_allclose(readout.std(), 1, rtol=1e-2)


def test_sum_superpixels():
    n_superpixels = 2
    n_samples = 10
    continuous_readout = np.ones((5, n_samples)) * np.arange(1, 6)[:, None]
    superpixel = np.array([0, 0, 1, 0, 1])
    superpixel_sum = sum_superpixels(
        continuous_readout, superpixel, n_superpixels
    )
    assert superpixel_sum.shape == (n_superpixels, n_samples)
    assert np.array_equal(superpixel_sum[0], np.full(10, 7))
    assert np.array_equal(superpixel_sum[1], np.full(10, 8))


def test_add_coincidence_window(acquisition):
    n_pixels = acquisition.camera.pixel.n_pixels
    time_axis = acquisition.camera.continuous_time_axis
    n_samples = time_axis.size
    division = acquisition.camera.continuous_sample_division
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


def test_get_digital_trigger_readout(acquisition):
    n_pixels = acquisition.camera.pixel.n_pixels
    n_superpixels = acquisition.camera.superpixel.n_superpixels
    time_axis = acquisition.camera.continuous_time_axis
    n_samples = time_axis.size
    division = acquisition.camera.continuous_sample_division
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
    trigger_readout = acquisition.get_digital_trigger_readout(continuous_readout)
    assert np.array_equal(trigger_readout, trigger_readout_expected)


def test_get_n_superpixel_triggers(acquisition):
    n_superpixels = 2
    n_samples = acquisition.camera.continuous_time_axis.size
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
    n_superpixels = camera.superpixel.n_superpixels
    n_samples = camera.continuous_time_axis.size
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_time, trigger_pair = acquisition.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)

    # No overlap (after sampling)
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_readout[0, 10:20] = True
    trigger_readout[0, 100:120] = True
    trigger_readout[1, 115:130] = True
    trigger_readout[2, 200:220] = True
    trigger_readout[3, 215:230] = True
    trigger_time, trigger_pair = acquisition.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)

    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_readout[0, 10:20] = True
    trigger_readout[0, 100:125] = True
    trigger_readout[1, 115:130] = True
    trigger_readout[2, 200:225] = True
    trigger_readout[3, 215:230] = True
    trigger_time, trigger_pair = acquisition.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (2,)
    assert trigger_pair.shape == (2, 2)
    assert np.array_equal(trigger_time, np.array([11, 21]))
    assert np.array_equal(trigger_pair, np.array([[0, 1], [2, 3]]))

    # Single pixel
    camera = Camera(pixel=PixelMapping(n_pixels=1))
    acquisition = EventAcquisition(camera=camera)
    n_superpixels = camera.superpixel.n_superpixels
    n_samples = camera.continuous_time_axis.size
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=np.bool)
    trigger_readout[0, 10:20] = True
    trigger_readout[0, 100:125] = True
    trigger_time, trigger_pair = acquisition.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)


def test_sample_waveform():
    camera = Camera()
    acquisition = EventAcquisition(camera=camera)
    n_pixels = camera.pixel.n_pixels
    time_axis = camera.continuous_time_axis
    n_continuous_samples = time_axis.size
    n_samples = camera.waveform_length * camera.sample_width
    continuous_readout = np.zeros((n_pixels, n_continuous_samples))
    continuous_readout[0, 300:305] = 100
    continuous_readout[2, 400:410] = 100

    waveform = acquisition.sample_readout(continuous_readout)
    assert waveform.shape == (n_pixels, n_samples)
    waveform_expected = np.zeros((n_pixels, n_samples))
    waveform_expected[0, 30] = 50
    waveform_expected[2, 40] = 100
    assert np.array_equal(waveform, waveform_expected)

    waveform = acquisition.sample_readout(continuous_readout, 30)
    assert waveform.shape == (n_pixels, n_samples)
    waveform_expected = np.zeros((n_pixels, n_samples))
    waveform_expected[0, 20] = 50
    waveform_expected[2, 30] = 100
    assert np.array_equal(waveform, waveform_expected)

    waveform = acquisition.sample_readout(continuous_readout, 25)
    assert waveform.shape == (n_pixels, n_samples)
    waveform_expected = np.zeros((n_pixels, n_samples))
    waveform_expected[0, 25] = 50
    waveform_expected[2, 35] = 100
    assert np.array_equal(waveform, waveform_expected)

    # Out of bounds
    with pytest.raises(ValueError):
        acquisition.sample_readout(continuous_readout, 10)
    with pytest.raises(ValueError):
        acquisition.sample_readout(continuous_readout, 900)

    # Single Pixel
    camera = Camera(pixel=PixelMapping(n_pixels=1))
    acquisition = EventAcquisition(camera=camera)
    n_pixels = camera.pixel.n_pixels
    time_axis = camera.continuous_time_axis
    n_continuous_samples = time_axis.size
    n_samples = camera.waveform_length * camera.sample_width
    continuous_readout = np.zeros((n_pixels, n_continuous_samples))
    continuous_readout[0, 300:305] = 100

    waveform = acquisition.sample_readout(continuous_readout)
    assert waveform.shape == (n_pixels, n_samples)
    waveform_expected = np.zeros((n_pixels, n_samples))
    waveform_expected[0, 30] = 50
    assert np.array_equal(waveform, waveform_expected)

    waveform = acquisition.sample_readout(continuous_readout, 30)
    assert waveform.shape == (n_pixels, n_samples)
    waveform_expected = np.zeros((n_pixels, n_samples))
    waveform_expected[0, 20] = 50
    assert np.array_equal(waveform, waveform_expected)
