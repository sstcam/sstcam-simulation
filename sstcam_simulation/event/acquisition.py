from ..camera import Camera
from .trigger import Trigger, NNSuperpixelAboveThreshold
from .photoelectrons import Photoelectrons
import numpy as np
from scipy.ndimage import convolve1d

__all__ = ["EventAcquisition"]


class EventAcquisition:
    def __init__(self, camera, trigger=None, seed=None):
        """
        Collection of methods which simulate operations performed by the camera
        electronics for event acquisition (e.g. sampling, trigger,
        digitisation), utilising the definitions within the camera container,
        and taking in the photoelectron container from the PhotoelectronSource
        as input.

        Parameters
        ----------
        camera : Camera
            Description of the camera
        trigger : Trigger
            Description of the trigger logic
            Default: NNSuperpixelAboveThreshold
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self.camera = camera
        self.seed = seed

        self.trigger = trigger
        if self.trigger is None:
            self.trigger = NNSuperpixelAboveThreshold(camera)

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

    def get_continuous_readout(self, photoelectrons):
        """
        Obtain the sudo-continuous readout from the camera for the given
        photoelectrons (signal and background) in this event.

        This is built by convolving the reference pulse shape of the camera
        with the arrival times and charge of the photoelectrons provided.
        Electronic noise is also included at this stage.

        The integral of this readout provides the total charge of the
        photoelectrons that arrived during the readout (in p.e. units).

        Parameters
        ----------
        photoelectrons : Photoelectrons
            Container for the photoelectron arrays, obtained from the EventSimulator

        Returns
        -------
        convolved : ndarray
            Array emulating continuous readout from the camera, with the
            photoelectrons convolved with the reference pulse shape
            Units: photoelectrons / ns
            Shape: (n_pixels, n_continuous_readout_samples)
        """
        # Samples corresponding to the photoelectron time
        time = photoelectrons.time
        sample = (time / self.camera.continuous_readout_sample_width).astype(np.int)

        # Add photoelectrons to the readout array
        pixel = photoelectrons.pixel
        charge = photoelectrons.charge
        n_samples = self.camera.continuous_readout_time_axis.size
        continuous_readout = np.zeros((self.camera.mapping.n_pixels, n_samples))
        np.add.at(continuous_readout, (pixel, sample), charge)

        # Convolve with the reference pulse shape
        #  TODO: remove bottleneck
        pulse = self.camera.reference_pulse.pulse
        origin = self.camera.reference_pulse.origin
        convolved = convolve1d(continuous_readout, pulse, mode="constant", origin=origin)

        # Add electronic noise
        noisy = self.camera.readout_noise.add_to_readout(convolved)

        return noisy

    def get_trigger(self, continuous_readout):
        """
        Get the triggers generated on the backplane as defined by the Trigger
        class

        Parameters
        ----------
        continuous_readout : ndarray
            Array emulating continuous readout from the camera
            Shape: (n_pixels, n_continuous_readout_samples)

        Returns
        -------
        trigger_time : ndarray
            Time of coincident rising edges between neighbouring superpixels (ns)
            Shape: (n_triggers)
        trigger_pair : ndarray
            The two neighbouring superpixels with coincident digital trigger readouts
            Shape: (n_triggers, 2)
        """
        return self.trigger(continuous_readout)

    def get_sampled_waveform(self, continuous_readout, trigger_time=None):
        """
        Sample the continuous readout by integrating over nanosecond bin
        widths, to produce a sampled waveform.

        The sum of all samples in the waveform provides the total charge that
        occurred within the waveform's duration (in p.e. units).

        Parameters
        ----------
        continuous_readout : ndarray
            Array emulating continuous readout from the camera
            Shape: (n_pixels, n_continuous_readout_samples)
        trigger_time : float
            Time of trigger. Start of waveform is dictated by this time minus
            the lookback time. If None (default), then the start of the readout
            is used as the waveform start.

        Returns
        -------
        waveform : ndarray
            Sampled waveform
            Units: photoelectrons
            Shape: (n_pixels, n_samples)
        """
        # Define start and end of waveform
        division = self.camera.continuous_readout_sample_division
        lookback_time = self.camera.lookback_time
        if trigger_time is None:
            start_time = 0
        else:
            start_time = trigger_time - lookback_time
        start = self.camera.get_continuous_readout_sample_from_time(start_time)
        end = int(start + self.camera.n_waveform_samples * division)

        if start < 0:
            raise ValueError("Digitisation begins before start of readout")
        if end > continuous_readout.shape[-1]:
            raise ValueError("Digitisation finishes after end of readout")
        readout_slice = continuous_readout[:, start:end]

        # Sum readout into samples
        division = self.camera.continuous_readout_sample_division
        n_pixels, n_readout_samples = readout_slice.shape
        n_samples = n_readout_samples // division
        waveform = readout_slice.reshape(
            (n_pixels, n_samples, division)
        ).sum(-1) * self.camera.continuous_readout_sample_width

        # Add electronic (digitisation) noise
        waveform = self.camera.digitisation_noise.add_to_readout(waveform)

        return waveform
