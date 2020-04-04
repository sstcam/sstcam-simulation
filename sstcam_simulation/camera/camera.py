from dataclasses import dataclass
from .pulse import ReferencePulse, GaussianPulse
from .spe import SPESpectrum, SiPMGentileSPE
from .constants import SAMPLE_WIDTH, SUB_SAMPLE_WIDTH


@dataclass
class Camera:
    """
    Container for properties which define the camera
    """
    n_pixels: int = 1
    reference_pulse: ReferencePulse = GaussianPulse()
    single_photoelectron_spectrum: SPESpectrum = SiPMGentileSPE()

    @property
    def sample_width(self):
        """Read-only. Unit: nanosecond"""
        return SAMPLE_WIDTH

    @property
    def sub_sample_width(self):
        """Read-only. Unit: nanosecond"""
        return SUB_SAMPLE_WIDTH
