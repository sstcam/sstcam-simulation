from .mapping import PixelMapping, SuperpixelMapping
from .pulse import ReferencePulse, GaussianPulse
from .spe import SPESpectrum, SiPMGentileSPE
from .constants import SAMPLE_WIDTH, CONTINUOUS_SAMPLE_DIVISION, CONTINUOUS_SAMPLE_WIDTH
from dataclasses import dataclass
import numpy as np


@dataclass
class Camera:
    """
    Container for properties which define the camera
    """
    continuous_readout_length: int = 1000  # Unit: nanosecond
    trigger_threshold: float = 0.5  # Unit: photoelectron / ns
    coincidence_window: float = 8  # Unit: ns
    pixel: PixelMapping = PixelMapping()
    superpixel: SuperpixelMapping = SuperpixelMapping(pixel=pixel)
    reference_pulse: ReferencePulse = GaussianPulse()
    photoelectron_spectrum: SPESpectrum = SiPMGentileSPE()

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
