from sstcam_simulation.event.trigger import (
    sum_superpixels, extend_digital_trigger, Trigger, NNSuperpixelAboveThreshold
)
from sstcam_simulation.camera import Camera, SSTCameraMapping
from sstcam_simulation.camera.spe import SiPMGentileSPE
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation import Photoelectrons, EventAcquisition
import numpy as np
import pytest


classes = Trigger.__subclasses__()


@pytest.mark.parametrize("trigger_class", classes)
def test_reference_pulses(trigger_class):
    camera = Camera()
    n_pixels = camera.mapping.n_pixels
    n_samples = camera.continuous_readout_time_axis.size
    trigger = trigger_class(camera)
    continuous_readout = np.zeros((n_pixels, n_samples))
    time = 10
    sample = camera.get_continuous_readout_sample_from_time(time)
    continuous_readout[:, sample] = 10000
    trigger_times, _ = trigger(continuous_readout)
    assert np.unique(trigger_times).size == 1
    assert trigger_times[0] == time


def test_sum_superpixels():
    n_superpixels = 2
    n_samples = 10
    continuous_readout = np.ones((5, n_samples)) * np.arange(1, 6)[:, None]
    superpixel = np.array([0, 0, 1, 0, 1])
    superpixel_sum = sum_superpixels(continuous_readout, superpixel, n_superpixels)
    assert superpixel_sum.shape == (n_superpixels, n_samples)
    assert np.array_equal(superpixel_sum[0], np.full(10, 7))
    assert np.array_equal(superpixel_sum[1], np.full(10, 8))


def test_extend_digital_trigger():
    camera = Camera(mapping=SSTCameraMapping(n_pixels=2))
    n_pixels = camera.mapping.n_pixels
    time_axis = camera.continuous_readout_time_axis
    n_samples = time_axis.size
    division = camera.continuous_readout_sample_division
    length = camera.digital_trigger_length * division
    above_threshold = np.zeros((n_pixels, n_samples), dtype=bool)
    above_threshold[0, 100:101] = True
    above_threshold[1, 200:500] = True
    above_threshold[1, 510:520] = True
    trigger_readout_expected = np.zeros((n_pixels, n_samples), dtype=bool)
    trigger_readout_expected[0, 100:101+length] = True
    trigger_readout_expected[1, 200:500+length] = True
    trigger_readout_expected[1, 510:520+length] = True
    trigger_readout = extend_digital_trigger(above_threshold, length)
    assert np.array_equal(trigger_readout, trigger_readout_expected)


def test_update_trigger_threshold():
    camera = Camera(trigger_threshold=5)
    trigger = NNSuperpixelAboveThreshold(camera=camera)
    assert trigger.camera.trigger_threshold == 5
    camera.update_trigger_threshold(6)
    assert trigger.camera.trigger_threshold == 6


def test_get_digital_trigger_readout():
    camera = Camera(mapping=SSTCameraMapping(n_pixels=2))
    trigger = NNSuperpixelAboveThreshold(camera=camera)
    n_pixels = trigger.camera.mapping.n_pixels
    n_superpixels = trigger.camera.mapping.n_superpixels
    time_axis = trigger.camera.continuous_readout_time_axis
    n_samples = time_axis.size
    division = trigger.camera.continuous_readout_sample_division
    length = trigger.camera.digital_trigger_length * division
    continuous_readout = np.zeros((n_pixels, n_samples))
    continuous_readout[0, 100:101] = 100
    continuous_readout[0, 200:500] = 0.1
    continuous_readout[1, 200:500] = 100
    continuous_readout[1, 510:520] = 100
    trigger_readout_expected = np.zeros((n_superpixels, n_samples), dtype=bool)
    trigger_readout_expected[0, 100] = True
    trigger_readout_expected[0, 200] = True
    trigger_readout_expected[0, 510] = True
    trigger_readout = trigger.get_superpixel_digital_trigger_line(continuous_readout)
    assert np.array_equal(trigger_readout, trigger_readout_expected)

    # Add window
    trigger_readout_expected = np.zeros((n_superpixels, n_samples), dtype=bool)
    trigger_readout_expected[0, 100:101+length] = True
    trigger_readout_expected[0, 200:201+length] = True
    trigger_readout_expected[0, 510:511+length] = True
    trigger_readout = trigger.extend_by_digital_trigger_length(trigger_readout)
    assert np.array_equal(trigger_readout, trigger_readout_expected)


def test_get_n_superpixel_triggers():
    camera = Camera(mapping=SSTCameraMapping(n_pixels=2))
    trigger = NNSuperpixelAboveThreshold(camera=camera)
    n_superpixels = 2
    n_samples = trigger.camera.continuous_readout_time_axis.size
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=bool)
    trigger_readout[0, 10] = True
    trigger_readout[0, 15] = True
    trigger_readout[1, 100] = True
    trigger_readout[1, 150] = True
    trigger_readout[1, 200] = True
    n_triggers = trigger.get_n_superpixel_triggers(trigger_readout)
    assert np.array_equal(n_triggers, np.array([2, 3], dtype=int))


def test_get_backplane_trigger():
    camera = Camera()
    trigger = NNSuperpixelAboveThreshold(camera=camera)
    n_superpixels = camera.mapping.n_superpixels
    n_samples = camera.continuous_readout_time_axis.size
    csample = camera.get_continuous_readout_sample_from_time
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=bool)
    trigger_time, trigger_pair = trigger.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)

    # No overlap (after sampling)
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=bool)
    trigger_readout[0, csample(1.00):csample(2.00)] = True
    trigger_readout[0, csample(10.0):csample(12.0)] = True
    trigger_readout[1, csample(11.5):csample(13.0)] = True
    trigger_readout[2, csample(20.0):csample(22.0)] = True
    trigger_readout[3, csample(21.5):csample(23.0)] = True
    trigger_time, trigger_pair = trigger.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)

    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=bool)
    trigger_readout[0, csample(1.00):csample(2.00)] = True
    trigger_readout[0, csample(10.0):csample(12.5)] = True
    trigger_readout[1, csample(11.5):csample(13.0)] = True
    trigger_readout[2, csample(20.0):csample(22.5)] = True
    trigger_readout[3, csample(21.5):csample(23.0)] = True
    trigger_time, trigger_pair = trigger.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (2,)
    assert trigger_pair.shape == (2, 2)
    assert np.array_equal(trigger_time, np.array([12, 22]))
    assert np.array_equal(trigger_pair, np.array([[0, 1], [2, 3]]))

    # Single pixel
    camera = Camera(mapping=SSTCameraMapping(n_pixels=1))
    trigger = NNSuperpixelAboveThreshold(camera=camera)
    n_superpixels = camera.mapping.n_superpixels
    n_samples = camera.continuous_readout_time_axis.size
    trigger_readout = np.zeros((n_superpixels, n_samples), dtype=bool)
    trigger_readout[0, 10:20] = True
    trigger_readout[0, 100:125] = True
    trigger_time, trigger_pair = trigger.get_backplane_trigger(trigger_readout)
    assert trigger_time.shape == (0,)
    assert trigger_pair.shape == (0, 2)


def test_trigger_threshold_units():
    def get_n_triggers_for_threshold(mv_per_pe, normalise_charge, trigger_threshold):
        camera = Camera(
            mapping=SSTCameraMapping(n_pixels=1),
            continuous_readout_duration=128,
            photoelectron_pulse=GaussianPulse(mv_per_pe=mv_per_pe),
            photoelectron_spectrum=SiPMGentileSPE(normalise_charge=normalise_charge),
            trigger_threshold=trigger_threshold
        )
        avg = camera.photoelectron_spectrum.average
        pe = Photoelectrons(np.array([0]), np.array([20]), np.array([avg]))
        acquisition = EventAcquisition(camera=camera)
        trigger = acquisition.trigger
        readout = acquisition.get_continuous_readout(pe)
        digital_trigger = trigger.get_superpixel_digital_trigger_line(readout)
        return trigger.get_n_superpixel_triggers(digital_trigger)
    n_triggers = get_n_triggers_for_threshold(None, True, 0.9)
    assert n_triggers == 1
    n_triggers = get_n_triggers_for_threshold(None, True, 1.1)
    assert n_triggers == 0
    n_triggers = get_n_triggers_for_threshold(None, False, 0.9)
    assert n_triggers == 1
    n_triggers = get_n_triggers_for_threshold(None, False, 1.1)
    assert n_triggers == 0
    n_triggers = get_n_triggers_for_threshold(4, True, 0.9)
    assert n_triggers == 1
    n_triggers = get_n_triggers_for_threshold(4, True, 1.1)
    assert n_triggers == 0
    n_triggers = get_n_triggers_for_threshold(4, False, 0.9)
    assert n_triggers == 1
    n_triggers = get_n_triggers_for_threshold(4, False, 1.1)
    assert n_triggers == 0
