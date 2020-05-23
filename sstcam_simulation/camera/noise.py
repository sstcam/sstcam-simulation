from abc import ABCMeta, abstractmethod
import numpy as np

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
