from sstcam_simulation.camera.constants import CONTINUOUS_READOUT_SAMPLE_WIDTH
from abc import ABCMeta, abstractmethod
from scipy import signal

__all__ = [
    "Coupling",
    "NoCoupling",
    "ACFilterCoupling",
    "ACOffsetCoupling",
]


class Coupling(metaclass=ABCMeta):
    """
    Base for classes which define the readout coupling, and how it is
    applied to the waveform
    """

    @abstractmethod
    def apply_to_readout(self, readout):
        """
        Apply the coupling to readout

        Parameters
        ----------
        readout : ndarray
            Array emulating continuous readout from the camera, with the
            photoelectrons convolved with the reference pulse shape
            Shape: (n_pixels, n_continuous_readout_samples)

        Returns
        -------
        coupled : ndarray
            Shape: (n_pixels, n_continuous_readout_samples)
        """


class NoCoupling(Coupling):
    """
    Apply no coupling to the readout
    """
    def apply_to_readout(self, readout):
        return readout


class ACFilterCoupling(Coupling):
    def __init__(self, order=1, critical_frequency=160e3):
        """
        Apply AC coupling in the form of a butterworth highpass filter

        This requires a large readout length for the baseline to reach a stable offset

        Parameters
        ----------
        order : int
            The order of the filter.
        critical_frequency : float
            The point at which the gain of the filter drops to 1/sqrt(2) that
            of the passband (the “-3 dB point”).
        """
        self.filter = signal.butter(
            order,
            critical_frequency,
            btype='highpass',
            fs=1/(CONTINUOUS_READOUT_SAMPLE_WIDTH * 1e-9),
            output='sos'
        )

    def apply_to_readout(self, readout):
        filtered = signal.sosfilt(self.filter, readout)
        return filtered


class ACOffsetCoupling(Coupling):
    def __init__(self, nsb_rate=40., pulse_area=1., spectrum_average=1.):
        """
        Apply AC Coupling in the form of an offset such that the baseline
        average is zero (in the presence of NSB)

        This essentially emulates a perfect AC coupling behaviour

        Parameters
        ----------
        nsb_rate : float
            The nsb or dark count photoelectron rate
            Units : MHz
        pulse_area : float
            Area of the 1 p.e. pulse (accessible from `camera.photoelectron_pulse.area`)
            Units : photoelectrons or mV * ns
        spectrum_average : float
            Average charge measured for 1 initial photoelectron
        """
        self._nsb_rate = nsb_rate
        self._pulse_area = pulse_area
        self._spectrum_average = spectrum_average

    @property
    def offset(self):
        return self._nsb_rate * 1e6 * self._pulse_area * 1e-9 * self._spectrum_average

    def update_nsb_rate(self, nsb_rate):
        self._nsb_rate = nsb_rate

    def apply_to_readout(self, readout):
        return readout - self.offset
