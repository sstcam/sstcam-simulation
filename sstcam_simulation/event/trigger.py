from abc import ABCMeta, abstractmethod
from numba import guvectorize, float64, boolean
import numpy as np

__all__ = [
    "sum_superpixels",
    "add_coincidence_window",
    "Trigger",
    "NNSuperpixelAboveThreshold",
]


def sum_superpixels(continuous_readout, pixel_to_superpixel, n_superpixels):
    """
    Sum the readouts from pixels of the same superpixel

    Parameters
    ----------
    continuous_readout : ndarray
        Readout from each pixel
        Shape: (n_pixels, n_continuous_readout_samples)
    pixel_to_superpixel : ndarray
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
    np.add.at(superpixel_sum, pixel_to_superpixel, continuous_readout)
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


class Trigger(metaclass=ABCMeta):
    def __init__(self, camera):
        """
        Base for classes which define the trigger logic for the camera.
        Subclasses should define the __call__ method to return the
        trigger times.
        """
        self.camera = camera

    @abstractmethod
    def __call__(self, continuous_readout):
        """
        Execute the trigger logic to find any triggers within the
        continuous readout.

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
        """


class NNSuperpixelAboveThreshold(Trigger):
    """
    Triggers are created when two neighbouring superpixels are above a
    threshold (within a coincidence window)
    """

    def __call__(self, continuous_readout):
        line = self.get_superpixel_digital_trigger_line(continuous_readout)
        extended = self.extend_by_coincidence_window(line)
        return self.get_backplane_trigger(extended)

    def get_superpixel_digital_trigger_line(self, continuous_readout):
        """
        Obtain the boolean digital trigger line for each superpixel, based on
        if the continuous readout (summed across each superpixel) is above the
        trigger threshold

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
        pixel_to_superpixel = self.camera.mapping.pixel_to_superpixel
        n_superpixels = self.camera.mapping.n_superpixels
        superpixel_sum = sum_superpixels(
            continuous_readout, pixel_to_superpixel, n_superpixels
        )

        # Discriminate superpixel readout with threshold
        # (First convert threshold to sample units, i.e. p.e./ns)
        sample_unit_per_photoelectron = self.camera.reference_pulse.peak_height
        threshold = self.camera.trigger_threshold * sample_unit_per_photoelectron
        above_threshold = superpixel_sum >= threshold

        return above_threshold

    def extend_by_coincidence_window(self, digital_trigger_line):
        """
        Extend the superpixel digital trigger line (boolean array) to
        account for the coincidence window

        Parameters
        ----------
        digital_trigger_line : ndarray
            Boolean array indicating where each superpixel line is "high" (True)
            Shape: (n_superpixels, n_continuous_readout_samples)

        Returns
        -------
        digital_trigger_line : ndarray
            Digital trigger line extended by the coincidence window
            Shape: (n_superpixels, n_continuous_readout_samples)
        """
        division = self.camera.continuous_readout_sample_division
        coincidence_samples = self.camera.coincidence_window * division
        return add_coincidence_window(digital_trigger_line, coincidence_samples)

    def get_backplane_trigger(self, digital_trigger_line, return_pairs=False):
        """
        Get the triggers generated on the backplane by looking for coincidences
        in the digital trigger line from neighbouring superpixels

        Parameters
        ----------
        digital_trigger_line : ndarray
            Boolean array indicating where each superpixel line is "high" (True)
            Shape: (n_superpixels, n_continuous_readout_samples)
        return_pairs : bool
            In addition to the default returns, include the ndarray indicating
            the superpixel pairs that caused the trigger

        Returns
        -------
        trigger_time : ndarray
            Time of coincident rising edges between neighbouring superpixels (ns)
            Shape: (n_triggers)
        trigger_pair : ndarray
            OPTIONAL (return_pairs)
            The two neighbouring superpixels with coincident digital trigger readouts
            Shape: (n_triggers, 2)
        """
        # Backplane clocks in the trigger every nanosecond
        division = self.camera.continuous_readout_sample_division
        sampled = digital_trigger_line[:, ::division]

        # Find coincident high trigger lines between neighbouring superpixels
        neighbours = self.camera.mapping.superpixel.neighbours
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

        if return_pairs:
            return trigger_time, trigger_pair
        else:
            return trigger_time

    @staticmethod
    def get_n_superpixel_triggers(digital_trigger_line):
        """
        Count the number of rising-edge threshold-crossings in the
        superpixel digital trigger line

        Parameters
        ----------
        digital_trigger_line : ndarray
            Boolean array indicating where each superpixel line is "high" (True)
            Shape: (n_superpixels, n_continuous_readout_samples)

        Returns
        -------
        ndarray
            Number of triggers in the digital signal readout per superpixel
        """
        return np.sum(np.diff(digital_trigger_line.astype(np.int), prepend=0) == 1, axis=1)
