from sstcam_simulation import Photoelectrons
from sstcam_simulation.camera.spe import single_gaussian, sipm_gentile_spe, \
    SPESpectrum, optical_crosstalk_probability
from inspect import isabstract
import numpy as np
import pytest


def test_pmt():
    x = np.linspace(0, 3, 10000)
    pdf = single_gaussian(x, 0.2)
    avg = np.average(x, weights=pdf)
    var = np.average((x - avg) ** 2, weights=pdf)
    np.testing.assert_allclose(avg, 1, rtol=1e-5)
    np.testing.assert_allclose(var, 0.04, rtol=1e-5)


def test_optical_crosstalk_probability():
    k = 1
    assert optical_crosstalk_probability(k, 0.3) == 1-0.3

    k = np.arange(1, 250)
    assert optical_crosstalk_probability(k, 0.2).sum() == 1

    assert optical_crosstalk_probability(0, 0.3) == 0


def test_sipm_gentile():
    x = np.linspace(0, 10, 10000)
    pdf = sipm_gentile_spe(x, 0.2, 0.3)
    avg = np.average(x, weights=pdf)
    var = np.average((x - avg) ** 2, weights=pdf)
    np.testing.assert_allclose(avg, 1.428433, rtol=1e-5)
    np.testing.assert_allclose(var, 0.668078, rtol=1e-5)


@pytest.mark.parametrize("spectrum_class", SPESpectrum.__subclasses__())
def test_spe_spectra(spectrum_class):
    if isabstract(spectrum_class):
        return

    n_photoelectrons = 10000
    photoelectrons = Photoelectrons(
        pixel=np.zeros(n_photoelectrons),
        time=np.zeros(n_photoelectrons),
        charge=np.ones(n_photoelectrons),
        metadata=dict(test="test"),
    )

    rng = np.random.RandomState(seed=1)

    spectrum = spectrum_class(normalise_charge=True)
    result = spectrum.apply(photoelectrons, rng)
    mean = result.charge.mean()
    std = result.charge.std()
    assert result is not photoelectrons
    np.testing.assert_allclose(spectrum.average, 1, rtol=1e-3)
    np.testing.assert_allclose(mean, 1, rtol=1e-3)
    np.testing.assert_allclose(1+std**2, spectrum.excess_noise_factor, rtol=1e-3)

    spectrum = spectrum_class(normalise_charge=False)
    result = spectrum.apply(photoelectrons, rng)
    mean = result.charge.mean()
    std = result.charge.std()
    assert result is not photoelectrons
    np.testing.assert_allclose(mean, spectrum.average, rtol=1e-3)
    np.testing.assert_allclose(1+std**2, spectrum.excess_noise_factor, rtol=1e-3)
