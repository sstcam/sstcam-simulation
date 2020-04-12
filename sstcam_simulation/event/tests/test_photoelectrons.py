from sstcam_simulation.event import Photoelectrons
import numpy as np


def test_photoelectrons():
    pixel = np.array([0], dtype=np.int)
    time = np.array([30])
    charge = np.array([2])
    photoelectrons_1 = Photoelectrons(pixel, time, charge)

    pixel = np.array([1], dtype=np.int)
    time = np.array([35])
    charge = np.array([5])
    photoelectrons_2 = Photoelectrons(pixel, time, charge)

    combined = photoelectrons_1 + photoelectrons_2
    assert combined.pixel.size == 2
    assert combined.time.size == 2
    assert combined.charge.size == 2
    np.testing.assert_allclose(combined.charge, np.array([2, 5]))

    photoelectrons_3 = Photoelectrons(pixel, time, charge)
    assert photoelectrons_1 != photoelectrons_2
    assert photoelectrons_2 == photoelectrons_3


def test_get_photoelectrons_per_pixel():
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1]), time=np.array([30, 40]), charge=np.array([1.0, 2.0])
    )
    pe = photoelectrons.get_photoelectrons_per_pixel(n_pixels=2)
    assert np.array_equal(pe, np.array([1, 1]))


def test_get_charge_per_pixel():
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1]), time=np.array([30, 40]), charge=np.array([1.0, 2.0])
    )
    charge = photoelectrons.get_charge_per_pixel(n_pixels=2)
    assert np.array_equal(charge, np.array([1.0, 2.0]))


def test_get_average_time_per_pixel():
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 0, 1, 1]),
        time=np.array([30, 40, 40, 50]),
        charge=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    time = photoelectrons.get_average_time_per_pixel(n_pixels=2)
    assert np.array_equal(time, np.array([35.0, 45.0]))
