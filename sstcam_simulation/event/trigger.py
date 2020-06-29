from abc import ABCMeta, abstractmethod
from numba import guvectorize, float64, boolean
import numpy as np

__all__ = [
    "sum_superpixels",
    "extend_digital_trigger",
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
def extend_digital_trigger(array, length, digital_signal):
    """
    Pad the digital trigger signals with True

    Parameters
    ----------
    array : ndarray
        Boolean array of samples, with True indicating the start of digital
        trigger signals that are to be extended
        Shape: (n_superpixels, n_continuous_readout_samples)
    length : int
        Number of samples for the digital signal to be extended by
    digital_signal : ndarray
        Inplace return array, containing the extended trigger digital signal
        Shape: (n_superpixels, n_continuous_readout_samples)
    """
    # TODO: Only reset counter after reaching end
    counter = 0
    for isample in range(array.size):
        if array[isample]:
            digital_signal[isample] = True
            counter = length
        else:
            if counter > 0:
                digital_signal[isample] = True
                counter -= 1
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
        extended = self.extend_by_digital_trigger_length(line)
        return self.get_backplane_trigger(extended)

    def get_superpixel_digital_trigger_line(self, continuous_readout):
        """
        Obtain the boolean digital trigger line for each superpixel, based on
        if the continuous readout (summed across each superpixel) crosses the
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

        # Convert threshold to sample units, i.e. p.e./ns
        sample_unit_per_photoelectron = self.camera.reference_pulse.peak_height
        threshold = self.camera.trigger_threshold * sample_unit_per_photoelectron

        # Discriminate superpixel readout with threshold
        above_threshold = superpixel_sum >= threshold

        # Only consider times when threshold is crossed
        prepend = 1 if above_threshold.all() else 0
        crossings = np.diff(above_threshold.astype(np.int), prepend=prepend) == 1

        return crossings

    def extend_by_digital_trigger_length(self, digital_trigger_line):
        """
        Extend the superpixel digital trigger line (boolean array) to
        account for the coincidence window

        Parameters
        ----------
        digital_trigger_line : ndarray
            Boolean array of samples, with True indicating the start of digital
            trigger signals that are to be extended
            Shape: (n_superpixels, n_continuous_readout_samples)

        Returns
        -------
        digital_trigger_line : ndarray
            Extended digital trigger line
            Shape: (n_superpixels, n_continuous_readout_samples)
        """
        division = self.camera.continuous_readout_sample_division
        length = self.camera.digital_trigger_length * division
        return extend_digital_trigger(digital_trigger_line, length)

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
        return np.sum(np.diff(digital_trigger_line.astype(np.int)) == 1, axis=1)
