from .constants import CONTINUOUS_READOUT_SAMPLE_WIDTH
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('default', module='sstcam_simulation')

__all__ = [
    "PhotoelectronPulse",
    "ReferencePulse",
    "GenericPulse",
    "GaussianPulse",
]


class PhotoelectronPulse(metaclass=ABCMeta):
    def __init__(self, duration, mv_per_pe=None):
        """
        Base for classes which define the reference pulse shape for a single
        photoelectron.

        The pulse is evaluated on initialisation.
        For a new pulse to be defined, a new class should be initialised.

        Parameters
        ----------
        duration : int
            Duration of the reference pulse in nanoseconds
        mv_per_pe : float or None
            Height of a 1 photoelectron pulse in mV. If this is set, then the
            units of the waveform samples can be globally considered as mV
            instead of photoelectrons/sample.
            If this is None, then the height of a 1 photoelectron pulse is
            determined such that the pulse area is 1
        """
        self.time = np.arange(0, duration, CONTINUOUS_READOUT_SAMPLE_WIDTH)
        amplitude = self._function(self.time)

        # Normalise the pulse to the correct units
        self.mv_per_pe = mv_per_pe
        if mv_per_pe is None:
            self._y_scale = amplitude.sum() * CONTINUOUS_READOUT_SAMPLE_WIDTH
        else:
            self._y_scale = amplitude.max() / mv_per_pe
        self.amplitude = amplitude / self._y_scale

        self.origin = self.amplitude.argmax() - self.amplitude.size // 2
        self.height = self.amplitude.max()
        self.area = self.amplitude.sum() * CONTINUOUS_READOUT_SAMPLE_WIDTH

    @property
    def y_scale(self):
        msg = "y_scale is deprecated"
        warnings.warn(msg, DeprecationWarning)
        return self._y_scale

    @property
    def pulse(self):
        msg = "pulse attribute is deprecated, replaced by amplitude"
        warnings.warn(msg, DeprecationWarning)
        return self.amplitude

    @property
    def peak_height(self):
        msg = "peak_height attribute is deprecated, replaced by height"
        warnings.warn(msg, DeprecationWarning)
        return self.height

    @property
    def sample_width(self):
        return self.time[1] - self.time[0]

    @abstractmethod
    def _function(self, time):
        """
        Function that describes the reference pulse shape.

        Parameters
        ----------
        time : ndarray
            Time in ns to evaluate the pulse at

        Returns
        -------
        photoelectron_pulse : ndarray
            Y values of reference pulse at the requested times  (not-normalised)
        """


class ReferencePulse(PhotoelectronPulse):
    def __init__(self, duration):
        msg = "ReferencePulse class is deprecated, replaced by PhotoelectronPulse"
        warnings.warn(msg, DeprecationWarning)
        super().__init__(duration)

    @abstractmethod
    def _function(self, time):
        """
        Function that describes the reference pulse shape.

        Parameters
        ----------
        time : ndarray
            Time in ns to evaluate the pulse at

        Returns
        -------
        photoelectron_pulse : ndarray
            Y values of reference pulse at the requested times  (not-normalised)
        """


class GenericPulse(PhotoelectronPulse):
    def __init__(self, time, value, mv_per_pe=None):
        """
        Reference pulse defined in an array, sampled at a sub-ns level
        (the finer the sampling the better).

        Evaluation of this reference pulse is performed with a linear interpolation.

        Parameters
        ----------
        time : ndarray
            Time (ns) axis for the reference pulse
        value : ndarray
            Values of the reference pulse corresponding to the time array
        mv_per_pe : float or None
            Height of a 1 photoelectron pulse in mV. If this is set, then the
            units of the waveform samples can be globally considered as mV
            instead of photoelectrons/sample.
            If this is None, then the height of a 1 photoelectron pulse is
            determined such that the pulse area is 1
        """
        self.interp_time = time
        self.interp_value = value
        duration = np.round(time[-1])
        super().__init__(duration=duration, mv_per_pe=mv_per_pe)

    def _function(self, time):
        return np.interp(time, xp=self.interp_time, fp=self.interp_value)


class GaussianPulse(PhotoelectronPulse):
    def __init__(self, mean=10, sigma=3, duration=20, mv_per_pe=None):
        """
        Simple gaussian reference pulse

        Parameters
        ----------
        mean : u.Quantity[time]
            Time (ns) corresponding to centre of pulse
        sigma : u.Quantity[time]
            Standard deviation of pulse (ns)
        duration : int
            Length of the reference pulse in nanoseconds
        mv_per_pe : float or None
            Height of a 1 photoelectron pulse in mV. If this is set, then the
            units of the waveform samples can be globally considered as mV
            instead of photoelectrons/sample.
            If this is None, then the height of a 1 photoelectron pulse is
            determined such that the pulse area is 1
        """
        self.mean = mean
        self.sigma = sigma
        super().__init__(duration=duration, mv_per_pe=mv_per_pe)

    def _function(self, time):
        return norm.pdf(time, loc=self.mean, scale=self.sigma)
