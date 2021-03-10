from sstcam_simulation.photoelectrons import Photoelectrons
import numpy as np


def test_photoelectrons():
    pixel = np.array([0], dtype=int)
    time = np.array([30])
    charge = np.array([2])
    meta1 = dict(test1=2, test3=1)
    photoelectrons_1 = Photoelectrons(pixel, time, charge, metadata=meta1)

    pixel = np.array([1], dtype=int)
    time = np.array([35])
    charge = np.array([5])
    initial = np.array([False])
    meta2 = dict(test1=3, test2=4)
    photoelectrons_2 = Photoelectrons(pixel, time, charge, initial=initial, metadata=meta2)

    combined = photoelectrons_1 + photoelectrons_2
    assert combined.pixel.size == 2
    assert combined.time.size == 2
    assert combined.charge.size == 2
    assert combined.initial.size == 2
    np.testing.assert_allclose(combined.charge, np.array([2, 5]))
    np.testing.assert_allclose(combined.initial, np.array([True, False]))
    assert combined.metadata == {**meta1, **meta2}

    photoelectrons_3 = Photoelectrons(pixel, time, charge, metadata=dict(test=2))
    photoelectrons_4 = Photoelectrons(pixel, time, charge, metadata=dict(test=2))
    assert photoelectrons_1 != photoelectrons_2
    assert photoelectrons_3 == photoelectrons_4


def test_get_photoelectrons_per_pixel():
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1, 1]),
        time=np.array([30, 40, 50]),
        charge=np.array([1.0, 2.0, 3.0])
    )
    pe = photoelectrons.get_photoelectrons_per_pixel(n_pixels=2)
    assert np.array_equal(pe, np.array([1, 2]))

    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1, 1]),
        time=np.array([30, 40, 50]),
        charge=np.array([1.0, 2.0, 3.0]),
        initial=np.array([True, False, True])
    )
    pe = photoelectrons.get_photoelectrons_per_pixel(n_pixels=2)
    assert np.array_equal(pe, np.array([1, 1]))


def test_get_charge_per_pixel():
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1, 1]),
        time=np.array([30, 40, 50]),
        charge=np.array([1.0, 2.0, 3.0])
    )
    charge = photoelectrons.get_charge_per_pixel(n_pixels=2)
    assert np.array_equal(charge, np.array([1.0, 5.0]))

    photoelectrons = Photoelectrons(
        pixel=np.array([0, 1, 1]),
        time=np.array([30, 40, 50]),
        charge=np.array([1.0, 2.0, 3.0]),
        initial=np.array([True, False, True])
    )
    charge = photoelectrons.get_charge_per_pixel(n_pixels=2)
    assert np.array_equal(charge, np.array([1.0, 5.0]))


def test_get_average_time_per_pixel():
    photoelectrons = Photoelectrons(
        pixel=np.array([0, 0, 1, 1]),
        time=np.array([30, 40, 40, 50]),
        charge=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    time = photoelectrons.get_average_time_per_pixel(n_pixels=2)
    assert np.array_equal(time, np.array([35.0, 45.0]))


def test_empty():
    photoelectrons = Photoelectrons.empty()
    assert photoelectrons.pixel.size == 0
    assert photoelectrons.time.size == 0
    assert photoelectrons.charge.size == 0
    assert photoelectrons.initial.size == 0
    assert photoelectrons.pixel.dtype == int
