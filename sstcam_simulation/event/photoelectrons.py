from dataclasses import dataclass
import numpy as np

__all__ = [
    "Photoelectrons",
]


@dataclass
class Photoelectrons:
    """
    Container for the properties of the photoelectrons that occur in an event

    Attributes
    ----------
    pixel : ndarray
        Array specifying the pixel which contains each photoelectron
        Shape: (n_photoelectrons)
    time : ndarray
        Array specifying the time of arrival for each photoelectron
        Shape: (n_photoelectrons)
    charge : ndarray
        Array specifying the charge reported for each photoelectron
        Shape: (n_photoelectrons)
    """
    pixel: np.ndarray
    time: np.ndarray
    charge: np.ndarray

    def __add__(self, other):
        pixel = np.concatenate([self.pixel, other.pixel])
        time = np.concatenate([self.time, other.time])
        charge = np.concatenate([self.charge, other.charge])
        return Photoelectrons(pixel=pixel, time=time, charge=charge)

    def __eq__(self, other):
        return (
                np.array_equal(self.pixel, other.pixel)
                & np.array_equal(self.time, other.time)
                & np.array_equal(self.charge, other.charge)
        )
