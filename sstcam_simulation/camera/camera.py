from .mapping import SSTCameraMapping
from .pulse import PhotoelectronPulse, GaussianPulse, ReferencePulse
from .spe import SPESpectrum, SiPMGentileSPE
from .noise import ElectronicNoise, PerfectElectronics
from .constants import WAVEFORM_SAMPLE_WIDTH, \
    CONTINUOUS_READOUT_SAMPLE_DIVISION, CONTINUOUS_READOUT_SAMPLE_WIDTH
from dataclasses import dataclass, field
import numpy as np
import pickle
import warnings
warnings.filterwarnings('default', module='sstcam_simulation')


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
    photoelectron_pulse: PhotoelectronPulse = GaussianPulse()
    reference_pulse: PhotoelectronPulse = field(default=None, repr=False, init=False)  #deprecated
    photoelectron_spectrum: SPESpectrum = SiPMGentileSPE()
    readout_noise: ElectronicNoise = PerfectElectronics()
    digitisation_noise: ElectronicNoise = PerfectElectronics()

    @property
    def reference_pulse(self):
        msg = "reference_pulse is deprecated, replaced by photoelectron_pulse"
        warnings.warn(msg, DeprecationWarning)
        return self.photoelectron_pulse

    @reference_pulse.setter
    def reference_pulse(self, reference_pulse):
        if reference_pulse is not None and type(reference_pulse) is not property:
            msg = "reference_pulse is deprecated, replaced by photoelectron_pulse"
            warnings.warn(msg, DeprecationWarning)
            super().__setattr__('photoelectron_pulse', reference_pulse)

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

    def save(self, path):
        """
        Save the camera object to disk

        Parameters
        ----------
        path : str
        """
        with open(path, mode='wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        """
        Load a camera object from disk

        Parameters
        ----------
        path : str

        Returns
        -------
        Camera
        """
        with open(path, mode='rb') as file:
            return pickle.load(file)

    @property
    def ctapipe_subarray(self):
        from ctapipe.instrument import TelescopeDescription, SubarrayDescription, \
            CameraGeometry, CameraReadout, CameraDescription, OpticsDescription
        import astropy.units as u

        geom = CameraGeometry(
            "sstcam",
            self.mapping.pixel.i,
            u.Quantity(self.mapping.pixel.x, 'm'),
            u.Quantity(self.mapping.pixel.y, 'm'),
            u.Quantity(self.mapping.pixel.size, 'm')**2,
            'square'
        )

        readout = CameraReadout(
            "sstcam",
            u.Quantity(1/self.waveform_sample_width, "GHz"),
            self.photoelectron_pulse.amplitude[None, :],
            u.Quantity(self.photoelectron_pulse.sample_width, "ns")
        )

        camera = CameraDescription("sstcam", geom, readout)
        optics = OpticsDescription.from_name('SST-ASTRI')
        telescope = TelescopeDescription("SST", "SST", optics, camera)
        subarray = SubarrayDescription(
            'toy',
            tel_positions={1: [0, 0, 0] * u.m},
            tel_descriptions={1: telescope},
        )
        return subarray
