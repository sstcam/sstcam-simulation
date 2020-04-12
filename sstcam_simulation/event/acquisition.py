from ..camera import Camera
import numpy as np
from scipy.ndimage import convolve1d
from numba import guvectorize, float64, boolean


__all__ = ["sum_superpixels", "add_coincidence_window", "EventAcquisition"]


def sum_superpixels(continuous_readout, superpixel, n_superpixels):
    """
    Sum the readouts from pixels of the same superpixel

    Parameters
    ----------
    continuous_readout : ndarray
        Readout from each pixel
        Shape: (n_pixels, n_continuous_readout_samples)
    superpixel : ndarray
        Superpixel index for each pixel in the continuous_readout
    n_superpixels : int
        Number of superpixels in the camera

    Returns
    -------
    superpixel_sum : ndarray
        Summed readout per superpixel
        Shape: (n_superpixels, n_continuous_readout_samples)
    """
    n_continuous_readout_samples = continuous_readout.shape[1]
    superpixel_sum = np.zeros((n_superpixels, n_continuous_readout_samples))
    np.add.at(superpixel_sum, superpixel, continuous_readout)
    return superpixel_sum


@guvectorize([(boolean[:], float64, boolean[:])], "(n),()->(n)", fastmath=True)
def add_coincidence_window(above_threshold, coincidence_samples, digital_signal):
    """
    Pad the digital trigger signal with True after above_threshold to emulate
    the coincidence window

    Parameters
    ----------
    above_threshold : ndarray
        Boolean array of samples that are above threshold
        Shape: (n_superpixels, n_continuous_readout_samples)
    coincidence_samples : int
        Number of samples that correspond to the coincidence window length
    digital_signal : ndarray
        Inplace return array, containing the trigger digital signal
        Shape: (n_superpixels, n_continuous_readout_samples)
    """
    above_counter = 0
    for isample in range(above_threshold.size):
        if above_threshold[isample]:
            digital_signal[isample] = True
            above_counter = coincidence_samples
        else:
            if above_counter > 0:
                digital_signal[isample] = True
                above_counter -= 1
            else:
                digital_signal[isample] = False


class EventAcquisition:
    def __init__(self, camera, seed=None):
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
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self.camera = camera
        self.seed = seed

    def get_continuous_readout(self, photoelectrons):
        """
        Obtain the sudo-continuous readout from the camera for the given
        photoelectrons (signal and background) in this event

        Parameters
        ----------
        photoelectrons : Photoelectrons
            Container for the photoelectron arrays, obtained from the EventSimulator

        Returns
        -------
        convolved : ndarray
            Array emulating continuous readout from the camera, with the
            photoelectrons convolved with the reference pulse shape
            Shape: (n_pixels, n_continuous_readout_samples)
        """
        # Samples corresponding to the photoelectron time
        time = photoelectrons.time
        sample = (time / self.camera.continuous_sample_width).astype(np.int)

        # Add photoelectrons to the readout array
        pixel = photoelectrons.pixel
        charge = photoelectrons.charge
        n_samples = self.camera.continuous_time_axis.size
        continuous_readout = np.zeros((self.camera.pixel.n_pixels, n_samples))
        np.add.at(continuous_readout, (pixel, sample), charge)

        # Convolve with the reference pulse shape
        #  TODO: remove bottleneck
        pulse = self.camera.reference_pulse.pulse
        origin = self.camera.reference_pulse.origin
        convolved = convolve1d(continuous_readout, pulse, mode="constant", origin=origin)

        # Add electronic noise
        noisy = self.camera.electronic_noise.add_to_readout(convolved)

        return noisy

    def get_digital_trigger_readout(self, continuous_readout):
        """
        Obtain the digital trigger readout, based on if the continuous readout
        (summed across each superpixel) is above the trigger threshold, and
        extending the resulting boolean array to account for the coincidence
        window

        Parameters
        ----------
        continuous_readout : ndarray
            Array emulating continuous readout from the camera
            Shape: (n_pixels, n_continuous_readout_samples)

        Returns
        -------
        digital_trigger : ndarray
            Boolean array indicating where each superpixel line is "high" (True)
            Shape: (n_superpixels, n_continuous_readout_samples)
        """

        # Sum superpixel readouts
        superpixel = self.camera.pixel.superpixel
        n_superpixels = self.camera.superpixel.n_superpixels
        superpixel_sum = sum_superpixels(continuous_readout, superpixel, n_superpixels)

        # Discriminate superpixel readout with threshold
        # (First convert threshold to sample units, i.e. p.e./ns)
        sample_unit_per_photoelectron = self.camera.reference_pulse.peak_height
        threshold = self.camera.trigger_threshold * sample_unit_per_photoelectron
        above_threshold = superpixel_sum >= threshold

        # Extend by coincidence window length
        division = self.camera.continuous_sample_division
        coincidence_samples = self.camera.coincidence_window * division
        digital_trigger = add_coincidence_window(above_threshold, coincidence_samples)

        return digital_trigger

    @staticmethod
    def get_n_superpixel_triggers(digital_trigger_readout):
        """
        Count the number of rising-edge threshold-crossings in the
        digital trigger readout

        Parameters
        ----------
        digital_trigger_readout : ndarray
            Boolean array indicating where each superpixel line is "high" (True)
            Shape: (n_superpixels, n_continuous_readout_samples)

        Returns
        -------
        ndarray
            Number of triggers in the digital signal readout per superpixel
        """
        return np.sum(np.diff(digital_trigger_readout.astype(np.int)) == 1, axis=1)

    def get_backplane_trigger(self, digital_trigger_readout):
        """
        Get the triggers generated on the backplane by looking for coincidences
        in the digital trigger readout from neighbouring superpixels

        Parameters
        ----------
        digital_trigger_readout : ndarray
            Boolean array indicating where each superpixel line is "high" (True)
            Shape: (n_superpixels, n_continuous_readout_samples)

        Returns
        -------
        trigger_time : ndarray
            Time of coincident rising edges between neighbouring superpixels (ns)
            Shape: (n_triggers)
        trigger_pair : ndarray
            The two neighbouring superpixels with coincident digital trigger readouts
            Shape: (n_triggers, 2)
        """
        # Backplane clocks in the trigger every nanosecond
        division = self.camera.continuous_sample_division
        sampled = digital_trigger_readout[:, ::division]

        # Find coincident high trigger lines between neighbouring superpixels
        neighbours = self.camera.superpixel.neighbours
        neighbour_coincidence = sampled[neighbours[:, 0]] & sampled[neighbours[:, 1]]

        # Extract the rising edge time for the coincident superpixels
        trigger_where = np.where(
            np.diff(neighbour_coincidence.astype(np.int), axis=1) == 1
        )
        trigger_pair = neighbours[trigger_where[0]]
        trigger_time = trigger_where[1] + 1  # Plus 1 because of np.diff

        # Sort by time
        sort = np.argsort(trigger_time)
        trigger_pair = trigger_pair[sort]
        trigger_time = trigger_time[sort]

        return trigger_time, trigger_pair

    def get_sampled_waveform(self, continuous_readout, trigger_time=None):
        """
        Sample the continuous readout into a waveform

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
            Shape: (n_pixels, n_samples)
        """
        # Define start and end of waveform
        division = self.camera.continuous_sample_division
        lookback_time = self.camera.lookback_time
        start = 0 if trigger_time is None else int((trigger_time-lookback_time)*division)
        end = int(start + self.camera.waveform_length * division)
        if start < 0:
            raise ValueError("Digitisation begins before start of readout")
        if end > continuous_readout.shape[-1]:
            raise ValueError("Digitisation finishes after end of readout")
        readout_slice = continuous_readout[:, start:end]

        # Sum readout into samples
        division = self.camera.continuous_sample_division
        n_pixels, n_readout_samples = readout_slice.shape
        n_samples = n_readout_samples // division
        waveform = readout_slice.reshape(
            (n_pixels, n_samples, division)
        ).sum(-1) / division

        return waveform
