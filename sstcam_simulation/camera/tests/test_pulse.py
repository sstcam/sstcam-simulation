from sstcam_simulation.camera.pulse import PhotoelectronPulse, GenericPulse, \
    ReferencePulse
from sstcam_simulation.camera.constants import CONTINUOUS_READOUT_SAMPLE_WIDTH
import numpy as np
from scipy.stats import norm
import pytest


classes = PhotoelectronPulse.__subclasses__()
# GenericPulse must be tested separately
classes.remove(GenericPulse)

# TODO: remove post-deprecation completion
classes.remove(ReferencePulse)


@pytest.mark.parametrize("ref_pulse_class", classes)
def test_reference_pulses(ref_pulse_class):
    pulse = ref_pulse_class()
    np.testing.assert_allclose(pulse.area, 1)
    assert pulse.time.size == pulse.amplitude.size


@pytest.mark.parametrize("ref_pulse_class", classes)
def test_reference_pulses_mv_per_pe(ref_pulse_class):
    pulse = ref_pulse_class(mv_per_pe=2)
    np.testing.assert_allclose(pulse.height, 2)
    assert pulse.time.size == pulse.amplitude.size


def test_generic_pulse():
    mean = 30
    sigma = 6
    input_time = np.linspace(0, 60, 300)
    input_values = norm.pdf(input_time, mean, sigma)
    pulse = GenericPulse(input_time, input_values)
    np.testing.assert_allclose(pulse.area, 1)
    assert pulse.time.size == pulse.amplitude.size
    assert pulse.amplitude.size == 60 / CONTINUOUS_READOUT_SAMPLE_WIDTH
