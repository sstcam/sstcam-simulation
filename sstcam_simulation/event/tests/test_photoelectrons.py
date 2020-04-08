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
