import numpy as np
from ..camera import Camera
from .photoelectrons import Photoelectrons

__all__ = [
    "PhotoelectronSource"
]


class PhotoelectronSource:
    def __init__(self, camera, seed=None):
        """
        Collection of methods which simulate illumination sources and the
        detection of the photons by the photosensors, providing photoelectron
        arrays

        Parameters
        ----------
        camera : Camera
            Description of the camera
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self.camera = camera
        self.rng = np.random.default_rng(seed=seed)

    @property
    def seed(self):
        return self.rng.bit_generator

    def get_nsb(self, rate):
        """
        Obtain the photoelectron arrays for random Night-Sky Background light

        Parameters
        ----------
        rate : float
            NSB rate in MHz (number of photoelectrons per microsecond)
            This is the rate after already accounting for Photon Detection Efficiency

        Returns
        -------
        Photoelectrons
            Container for the NSB photoelectron arrays
        """
        length = self.camera.continuous_readout_length
        n_pixels = self.camera.pixel.n_pixels
        time_axis = self.camera.continuous_time_axis
        spectrum = self.camera.photoelectron_spectrum

        # Number of NSB photoelectrons per pixel in this event
        avg_photons_per_waveform = rate * 1e6 * length * 1e-9
        n_nsb_per_pixel = self.rng.poisson(avg_photons_per_waveform, n_pixels)
        print(n_nsb_per_pixel)

        # Pixel containing each photoelectron
        nsb_pixel = np.repeat(np.arange(n_pixels), n_nsb_per_pixel)

        # Uniformly distribute NSB photoelectrons in time across waveform
        n_photoelectrons = nsb_pixel.size
        nsb_time = self.rng.choice(time_axis, size=n_photoelectrons)

        # Get the charge reported by the photosensor (Inverse Transform Sampling)
        nsb_charge = self.rng.choice(spectrum.x, size=n_photoelectrons, p=spectrum.pdf)

        return Photoelectrons(pixel=nsb_pixel, time=nsb_time, charge=nsb_charge)

    # def get_cherenkov_shower(self):
    #     pass
    #     # return charge and time
    #
    # def get_uniform_illumination(self):
    #     pass
    #     # return charge and time
