from .mapping import SSTCameraMapping
from .pulse import ReferencePulse, GaussianPulse
from .spe import SPESpectrum, SiPMGentileSPE
from .noise import ElectronicNoise, PerfectElectronics
from .constants import WAVEFORM_SAMPLE_WIDTH, \
    CONTINUOUS_READOUT_SAMPLE_DIVISION, CONTINUOUS_READOUT_SAMPLE_WIDTH
from dataclasses import dataclass
import numpy as np

__all__ = [
    "Camera",
]


@dataclass(frozen=True)
class Camera:
    """
    Container for properties which define the camera
    """

    continuous_readout_duration: int = 1000  # Unit: nanosecond
    n_waveform_samples: int = 128
    trigger_threshold: float = 2  # Unit: photoelectron
    digital_trigger_length: float = 8  # Unit: nanosecond
    lookback_time: float = 20  # Unit: nanosecond
    mapping: SSTCameraMapping = SSTCameraMapping()
    reference_pulse: ReferencePulse = GaussianPulse()
    photoelectron_spectrum: SPESpectrum = SiPMGentileSPE()
    readout_noise: ElectronicNoise = PerfectElectronics()
    digitisation_noise: ElectronicNoise = PerfectElectronics()

    @property
    def waveform_sample_width(self):
        """Read-only. Unit: nanosecond"""
        return WAVEFORM_SAMPLE_WIDTH

    @property
    def continuous_readout_sample_division(self):
        """Read-only"""
        return CONTINUOUS_READOUT_SAMPLE_DIVISION

    @property
    def continuous_readout_sample_width(self):
        """Read-only. Unit: nanosecond"""
        return CONTINUOUS_READOUT_SAMPLE_WIDTH

    @property
    def continuous_readout_time_axis(self):
        """Time axis for the continuous readout. Unit: nanosecond"""
        return np.arange(
            0, self.continuous_readout_duration, self.continuous_readout_sample_width
        )

    @property
    def waveform_duration(self):
        """Unit: nanosecond"""
        return self.n_waveform_samples / self.waveform_sample_width

    @property
    def waveform_time_axis(self):
        """Time axis for the waveform. Unit: nanosecond"""
        return np.arange(0, self.n_waveform_samples, self.waveform_sample_width)

    def get_waveform_sample_from_time(self, time):
        return int(time / self.waveform_sample_width)

    def get_continuous_readout_sample_from_time(self, time):
        return int(time / self.continuous_readout_sample_width)

    def update_trigger_threshold(self, trigger_threshold):
        super().__setattr__('trigger_threshold', trigger_threshold)
