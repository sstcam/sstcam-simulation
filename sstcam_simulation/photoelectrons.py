from dataclasses import dataclass, field
import numpy as np

__all__ = [
    "Photoelectrons",
]


@dataclass
class Photoelectrons:
    """
    Container for the photoelectron arrays. These are arrays which describe
    the pixel, arrival time, and reported charge of each photoelectron.

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
    initial : ndarray
        (Optional, default = numpy array of True)
        Array specifying if the photoelectron is included in the initial
        photoelectrons before opct considerations. Only relevant for the
        SPESpectra that add ADDITIONAL photoelectrons (i.e. NOT SiPMGentileSPE)
        Shape: (n_photoelectrons)
    metadata : dict
        Dict of metadata for the photoelectrons
    """

    pixel: np.ndarray
    time: np.ndarray
    charge: np.ndarray
    initial: np.ndarray = field(default=None)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.initial is None:
            self.initial = np.ones(self.pixel.size, dtype=bool)

    def __len__(self):
        return self.pixel.size

    def __add__(self, other):
        pixel = np.concatenate([self.pixel, other.pixel])
        time = np.concatenate([self.time, other.time])
        charge = np.concatenate([self.charge, other.charge])
        initial = np.concatenate([self.initial, other.initial])
        metadata = {**self.metadata, **other.metadata}
        return Photoelectrons(
            pixel=pixel, time=time, charge=charge, initial=initial, metadata=metadata
        )

    def __eq__(self, other: "Photoelectrons"):
        return (
            np.array_equal(self.pixel, other.pixel)
            & np.array_equal(self.time, other.time)
            & np.array_equal(self.charge, other.charge)
            & np.array_equal(self.initial, other.initial)
            & (self.metadata == other.metadata)
        )

    def get_time_slice(self, start, stop):
        within = (self.time > start) & (self.time < stop)
        return Photoelectrons(
            pixel=self.pixel[within],
            time=self.time[within],
            charge=self.charge[within],
            initial=self.initial[within],
            metadata=self.metadata
        )

    def get_photoelectrons_per_pixel(self, n_pixels):
        """Integer count of photoelectrons in each photosensor pixel"""
        pixel_photoelectrons = np.zeros(n_pixels, dtype=int)
        pixel = self.pixel[self.initial]
        np.add.at(pixel_photoelectrons, pixel, 1)
        return pixel_photoelectrons

    def get_charge_per_pixel(self, n_pixels):
        """
        Total sum of charge (in units of p.e.) reported by each photosensor pixel
        (resulting from the SPE spectrum)
        """
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

    @classmethod
    def empty(cls):
        return cls(np.empty(0, dtype=int), np.empty(0), np.empty(0))
