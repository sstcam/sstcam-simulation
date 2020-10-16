import numpy as np
from ..camera import Camera
from .photoelectrons import Photoelectrons
from .cherenkov import get_cherenkov_shower_image

__all__ = ["PhotoelectronSource"]


class PhotoelectronSource:
    def __init__(self, camera, seed=None):
        """
        Collection of methods which simulate illumination sources and the
        detection of the photons by the photosensors.

        Each method returns a :class:`Photoelectrons` object, which is a
        container of 1D arrays describing the pixel, arrival time, and
        reported charge of each photoelectron.

        Parameters
        ----------
        camera : Camera
            Description of the camera
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self.camera = camera
        self.seed = seed

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
        rng = np.random.default_rng(seed=self.seed)

        # Number of NSB photoelectrons per pixel in this event
        length = self.camera.continuous_readout_duration
        n_pixels = self.camera.mapping.n_pixels
        avg_photons_per_waveform = rate * 1e6 * length * 1e-9
        n_nsb_per_pixel = rng.poisson(avg_photons_per_waveform, n_pixels)

        # Pixel containing each photoelectron
        nsb_pixel = np.repeat(np.arange(n_pixels), n_nsb_per_pixel)

        # Uniformly distribute NSB photoelectrons in time across waveform
        n_photoelectrons = nsb_pixel.size
        duration = self.camera.continuous_readout_duration
        nsb_time = rng.uniform(0, duration, size=n_photoelectrons)

        # Get the charge reported by the photosensor (Inverse Transform Sampling)
        spectrum = self.camera.photoelectron_spectrum
        nsb_charge = rng.choice(spectrum.x, size=n_photoelectrons, p=spectrum.pdf)

        return Photoelectrons(pixel=nsb_pixel, time=nsb_time, charge=nsb_charge)

    def get_uniform_illumination(self, time, illumination, laser_pulse_width=0):
        """
        Simulate the camera being illuminated by a uniform light (which already
        accounts for the focal plane curvature).

        Parameters
        ----------
        time : float
            Arrival time of the light at the focal plane
        illumination : float
            Average illumination in number of photoelectrons
        laser_pulse_width : float
            Width of the pulse from the illumination source

        Returns
        -------
        Photoelectrons
            Container for the photoelectron arrays
        """
        rng = np.random.default_rng(seed=self.seed)

        # Poisson fluctuation of photoelectrons
        n_pixels = self.camera.mapping.n_pixels
        n_pe_per_pixel = rng.poisson(illumination, n_pixels)

        # Pixel containing each photoelectron
        pixel = np.repeat(np.arange(n_pixels), n_pe_per_pixel)

        # Time of arrival for each photoelectron
        n_photoelectrons = pixel.size
        time = rng.normal(time, laser_pulse_width, n_photoelectrons)

        # Get the charge reported by the photosensor (Inverse Transform Sampling)
        spectrum = self.camera.photoelectron_spectrum
        charge = rng.choice(spectrum.x, size=n_photoelectrons, p=spectrum.pdf)

        return Photoelectrons(pixel=pixel, time=time, charge=charge)

    def get_cherenkov_shower(
        self,
        centroid_x,
        centroid_y,
        length,
        width,
        psi,
        time_gradient,
        time_intercept,
        intensity,
        cherenkov_pulse_width=3,
    ):
        """
        Simulate a Cherenkov shower image

        Parameters
        ----------
        centroid_x : float
            X coordinate for the center of the ellipse. Unit: m
        centroid_y : float
            Y coordinate for the center of the ellipse. Unit: m
        length : float
            Length of the ellipse. Unit: m
        width : float
            Width of the ellipse. Unit: m
        psi : float
            Rotation of the ellipse major axis from the X axis. Unit: degrees
        time_gradient : float
            Rate at which the time changes with distance along the shower axis
            Unit: ns / m
        time_intercept : float
            Pulse time at the shower centroid. Unit: ns
        intensity : float
            Average total number of photoelectrons contained in shower image
        cherenkov_pulse_width : float
            Width of Cherenkov pulse in a single pixel. Unit: ns

        Returns
        -------
        Photoelectrons
            Container for the photoelectron arrays
        """
        rng = np.random.default_rng(seed=self.seed)

        image_pe_pdf, image_time = get_cherenkov_shower_image(
            self.camera.mapping.pixel.x,
            self.camera.mapping.pixel.y,
            centroid_x,
            centroid_y,
            length,
            width,
            psi,
            time_gradient,
            time_intercept,
        )

        # Obtain number of photoelectrons generated in each pixel
        n_pe_per_pixel = rng.poisson(image_pe_pdf * intensity)

        # Pixel containing each photoelectron
        n_pixels = self.camera.mapping.n_pixels
        pixel = np.repeat(np.arange(n_pixels), n_pe_per_pixel)

        # Time of arrival for each photoelectron
        time = rng.normal(np.repeat(image_time, n_pe_per_pixel), cherenkov_pulse_width)

        # Get the charge reported by the photosensor (Inverse Transform Sampling)
        spectrum = self.camera.photoelectron_spectrum
        n_photoelectrons = pixel.size
        charge = rng.choice(spectrum.x, size=n_photoelectrons, p=spectrum.pdf)

        return Photoelectrons(pixel=pixel, time=time, charge=charge)

    def get_random_cherenkov_shower(self, cherenkov_pulse_width=3):
        """
        Simulate a random Cherenkov shower image

        Parameters
        ----------
        cherenkov_pulse_width : float
            Width of Cherenkov pulse in a single pixel. Unit: ns

        Returns
        -------
        Photoelectrons
            Container for the photoelectron arrays
        """
        rng = np.random.default_rng(seed=self.seed)

        xpix = self.camera.mapping.pixel.x
        ypix = self.camera.mapping.pixel.y
        centroid_x = rng.uniform(xpix.min(), xpix.max())
        centroid_y = rng.uniform(ypix.min(), ypix.max())
        width = rng.uniform(0, 0.03)
        length = rng.uniform(width, 0.1)
        psi = rng.uniform(0, 360)
        time_gradient = rng.uniform(-20, 20)
        time_intercept = rng.uniform(0, self.camera.continuous_readout_duration)
        intensity = rng.uniform(100, 100000)
        return self.get_cherenkov_shower(
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            length=length,
            width=width,
            psi=psi,
            time_gradient=time_gradient,
            time_intercept=time_intercept,
            intensity=intensity,
            cherenkov_pulse_width=cherenkov_pulse_width,
        )

    def resample_photoelectron_charge(self, pe: Photoelectrons) -> Photoelectrons:
        """
        Resample the charges of the photoelectrons from the spectrum defined in
        the Camera

        Parameters
        ----------
        pe : Photoelectrons

        Returns
        -------
        Photoelectrons
        """
        rng = np.random.default_rng(seed=self.seed)
        spectrum = self.camera.photoelectron_spectrum
        n_photoelectrons = pe.pixel.size
        charge = rng.choice(spectrum.x, size=n_photoelectrons, p=spectrum.pdf)
        return Photoelectrons(
            pixel=pe.pixel, time=pe.time, charge=charge, metadata=pe.metadata
        )