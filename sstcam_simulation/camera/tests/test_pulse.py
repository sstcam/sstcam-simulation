from sstcam_simulation.camera.pulse import ReferencePulse, GenericPulse
from sstcam_simulation.camera.constants import SUB_SAMPLE_WIDTH
import numpy as np
from scipy.stats import norm
import pytest


classes = ReferencePulse.__subclasses__()
# GenericPulse must be tested separately
classes.remove(GenericPulse)


@pytest.mark.parametrize("ref_pulse_class", classes)
def test_reference_pulses(ref_pulse_class):
    reference_pulse = ref_pulse_class()
    np.testing.assert_allclose(reference_pulse.pulse.sum() * SUB_SAMPLE_WIDTH, 1)
    assert reference_pulse.time.size == reference_pulse.pulse.size


def test_generic_pulse():
    mean = 30
    sigma = 6
    input_time = np.linspace(0, 60, 300)
    input_values = norm.pdf(input_time, mean, sigma)
    reference_pulse = GenericPulse(input_time, input_values)
    np.testing.assert_allclose(reference_pulse.pulse.sum() * SUB_SAMPLE_WIDTH, 1)
    assert reference_pulse.time.size == reference_pulse.pulse.size
    assert reference_pulse.pulse.size == 600
