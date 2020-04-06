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

    @property
    def continuous_time_axis(self):
        """Time axis for the continuous readout. Unit: nanosecond"""
        return np.arange(0, self.waveform_length, CONTINUOUS_SAMPLE_WIDTH)

    def get_continuous_readout(self, pixel, time, charge):
        """
        Obtain the sudo-continuous readout from the camera for the given
        photoelectrons (signal and background) in this event

        Parameters
        ----------
        pixel : ndarray
            Array specifying the pixel which contains each photoelectron
            Shape: (n_photoelectrons)
        time : ndarray
            Array specifying the time of arrival for each photoelectron
            Shape: (n_photoelectrons)
        charge : ndarray
            Array specifying the charge reported for each photoelectron
            Shape: (n_photoelectrons)

        Returns
        -------
        convolved : ndarray
            Array emulating continuous readout from the camera, with the
            photoelectrons convolved with the reference pulse shape
            Shape: (n_pixels, n_continuous_readout_samples)
        """
        # Samples corresponding to the photoelectron time
        sample = (time / self.continuous_sample_width).astype(np.int)

        # Add photoelectrons to the readout array
        n_samples = self.continuous_time_axis.size
        continuous_readout = np.zeros((self.n_pixels, n_samples))
        np.add.at(continuous_readout, (pixel, sample), charge)

        # Convolve with the reference pulse shape
        #  TODO: remove bottleneck
        pulse = self.reference_pulse.pulse
        origin = self.reference_pulse.origin
        convolved = convolve1d(continuous_readout, pulse, mode="constant", origin=origin)
        return convolved
