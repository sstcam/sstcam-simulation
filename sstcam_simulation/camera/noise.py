from sstcam_simulation.data import get_data
from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.interpolate import interp1d

__all__ = [
    "ElectronicNoise",
    "PerfectElectronics",
    "GaussianNoise",
]


class ElectronicNoise(metaclass=ABCMeta):
    """
    Base for classes which define an electronic noise spectrum, and how it is
    applied to the waveform

    Can be used for any of the noise components
    """

    @abstractmethod
    def add_to_readout(self, readout):
        """
        Add the electronic noise to the readout

        Parameters
        ----------
        readout : ndarray
            Array emulating continuous readout from the camera, with the
            photoelectrons convolved with the reference pulse shape
            Shape: (n_pixels, n_continuous_readout_samples)

        Returns
        -------
        noisy_readout : ndarray
            Array emulating continuous readout from the camera, with the
            electronic noise included
            Shape: (n_pixels, n_continuous_readout_samples)
        """


class PerfectElectronics(ElectronicNoise):
    """
    Perfect readout electronics that do not add any noise
    """

    def add_to_readout(self, readout):
        return readout


class GaussianNoise(ElectronicNoise):
    def __init__(self, stddev=1, seed=None):
        """
        Fluctuate readout with Gaussian noise

        Parameters
        ----------
        stddev : float
            Standard deviation of the gaussian noise
            Units: photoelectrons / ns
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self.stddev = stddev
        self.seed = seed

    def add_to_readout(self, readout):
        rng = np.random.default_rng(seed=self.seed)
        return rng.normal(readout, self.stddev)


class TemplateNoise(ElectronicNoise):
    default_path = get_data("datasheet/noise_LMH6722_opamp.txt")

    def __init__(self, n_samples, sample_width, filepath=default_path, stddev=1, seed=None):
        """
        Noise defined by a template such as that from a datasheet

        Parameters
        ----------
        n_samples : int
            Number of samples in the readout/waveform
        sample_width : float
            Width of samples in the readout/waveform (ns)
        stddev : float
            Standard deviation of the noise
            Units: photoelectrons / ns
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self._n_samples = n_samples
        self._sample_width = sample_width
        self._filepath = filepath
        self._stddev = stddev
        self._seed = seed

        self._frequency, self._v_root = np.loadtxt(filepath, delimiter=',', unpack=True)

        # Find scaling for requested stddev
        n_samples_long = int(1e7)
        voltage = self.get_interpolated_voltage(n_samples_long, sample_width)
        frequency_spectrum = self.get_frequency_spectrum(voltage)
        noise = self.get_noise(frequency_spectrum, n_samples_long)
        self.scale = stddev / np.std(noise)

    def get_interpolated_voltage(self, n_samples, sample_width):
        df = np.fft.fftfreq(n_samples) / sample_width
        df_positive = df[:len(df)//2]
        delta_df_positive = df_positive[1] - df_positive[0]
        f = interp1d(self._frequency, self._v_root)
        frequency_min = np.min(self._frequency)
        frequency_max = np.max(self._frequency)
        frequency_range = frequency_max - frequency_min
        frequency_interp = np.arange(frequency_min, frequency_max, frequency_range/n_samples)
        v_root_interp = f(frequency_interp)
        return v_root_interp * np.sqrt(delta_df_positive)

    def get_frequency_spectrum(self, voltage):
        rng = np.random.default_rng(seed=self._seed)
        phi = rng.uniform(0, 2*np.pi, size=voltage.size)  # Randomising phi from 0 to 2pi
        cplx = np.zeros(voltage.size, dtype=complex)
        i = np.arange(1, voltage.size//2)
        cplx.real[i] = voltage[i]*np.cos(phi[i])
        cplx.imag[i] = -voltage[i]*np.sin(phi[i])
        cplx.real[-i] = voltage[i]*np.cos(phi[i])
        cplx.imag[-i] = voltage[i]*np.sin(phi[i])
        return cplx

    @staticmethod
    def get_noise(frequency_spectrum, n_samples):
        return np.fft.ifft(frequency_spectrum) * n_samples * 1e-9  # Convert to Volts

    @staticmethod
    def get_noise_envelope(noise, sample_len):
        """
        Return back to the noise envelope from the simulated noise

        Parameters
        ----------
        noise : ndarray
            Noise component of the waveform
        sample_len : int
            Number of samples in the readout

        Returns
        -------
        ndarray
        """
        spectrum = np.fft.fft(noise*1e9 / sample_len)  # Convert to nV and rescale for FFT
        return np.abs(spectrum)

    def add_to_readout(self, readout):
        voltage = self.get_interpolated_voltage(self._n_samples, self._sample_width)
        frequency_spectrum = self.get_frequency_spectrum(voltage)
        noise = self.get_noise(frequency_spectrum, self._n_samples)
        return readout + noise * self.scale
