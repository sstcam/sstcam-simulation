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

    def __len__(self):
        return self.pixel.size

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

    def get_photoelectrons_per_pixel(self, n_pixels):
        """Number of photoelectrons in each photosensor pixel"""
        pixel_photoelectrons = np.zeros(n_pixels, dtype=np.int)
        np.add.at(pixel_photoelectrons, self.pixel, 1)
        return pixel_photoelectrons

    def get_charge_per_pixel(self, n_pixels):
        """Total charge (in units of p.e.) reported by each photosensor pixel"""
        pixel_charge = np.zeros(n_pixels)
        np.add.at(pixel_charge, self.pixel, self.charge)
        return pixel_charge

    def get_average_time_per_pixel(self, n_pixels):
        """Average arrival time of photoelectrons per pixel"""
        sum_ = np.zeros(n_pixels)
        n = np.zeros(n_pixels)
        np.add.at(sum_, self.pixel, self.time)
        np.add.at(n, self.pixel, 1)
        return np.divide(sum_, n, out=np.full_like(sum_, np.nan), where=n != 0)
