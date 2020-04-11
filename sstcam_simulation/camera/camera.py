from .mapping import PixelMapping, SuperpixelMapping
from .pulse import ReferencePulse, GaussianPulse
from .spe import SPESpectrum, SiPMGentileSPE
from .noise import ElectronicNoise, PerfectElectronics
from .constants import SAMPLE_WIDTH, CONTINUOUS_SAMPLE_DIVISION, CONTINUOUS_SAMPLE_WIDTH
from dataclasses import dataclass, field
import numpy as np

__all__ = [
    "Camera",
]


@dataclass(frozen=True)
class Camera:
    """
    Container for properties which define the camera
    """
    continuous_readout_length: int = 1000  # Unit: nanosecond
    waveform_length: int = 128  # Unit: nanosecond
    trigger_threshold: float = 2  # Unit: photoelectron
    coincidence_window: float = 8  # Unit: ns
    lookback_time: float = 20  # Unit: ns
    pixel: PixelMapping = PixelMapping()
    superpixel: SuperpixelMapping = field(init=False)
    reference_pulse: ReferencePulse = GaussianPulse()
    photoelectron_spectrum: SPESpectrum = SiPMGentileSPE()
    electronic_noise: ElectronicNoise = PerfectElectronics()

    def __post_init__(self):
        super().__setattr__('superpixel', SuperpixelMapping(self.pixel))

    @property
    def sample_width(self):
        """Read-only. Unit: nanosecond"""
        return SAMPLE_WIDTH

    @property
    def continuous_sample_division(self):
        """Read-only"""
        return CONTINUOUS_SAMPLE_DIVISION

    @property
    def continuous_sample_width(self):
        """Read-only. Unit: nanosecond"""
        return CONTINUOUS_SAMPLE_WIDTH

    @property
    def continuous_time_axis(self):
        """Time axis for the continuous readout. Unit: nanosecond"""
        return np.arange(0, self.continuous_readout_length, self.continuous_sample_width)

    def get_waveform_sample_from_time(self, time):
        return int(time / self.sample_width)

    def get_continuous_readout_sample_from_time(self, time):
        return int(time / self.sample_width * self.continuous_sample_division)

    def update_trigger_threshold(self, trigger_threshold):
        super().__setattr__('trigger_threshold', trigger_threshold)
