from .pulse import ReferencePulse, GaussianPulse
from .spe import SPESpectrum, SiPMGentileSPE
from .constants import SAMPLE_WIDTH, CONTINUOUS_SAMPLE_WIDTH
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import convolve1d


@dataclass
class Camera:
    """
    Container for properties which define the camera
    """
    n_pixels: int = 1
    waveform_length: int = 128  # Unit: nanosecond
    reference_pulse: ReferencePulse = GaussianPulse()
    photoelectron_spectrum: SPESpectrum = SiPMGentileSPE()

    @property
    def sample_width(self):
        """Read-only. Unit: nanosecond"""
        return SAMPLE_WIDTH

    @property
    def continuous_sample_width(self):
        """Read-only. Unit: nanosecond"""
        return CONTINUOUS_SAMPLE_WIDTH
