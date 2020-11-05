from sstcam_simulation import Photoelectrons
from sstcam_simulation.camera.spe import single_gaussian, sipm_gentile_spe, \
    SPESpectrum, optical_crosstalk_probability, SiPMDelayed
from ctapipe.core import non_abstract_children
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


def _get_result_photoelectrons(spectrum, rng):
    n_events = 1000
    pe = []
    charge = []
    for iev in range(n_events):
        n_photoelectrons = 1
        photoelectrons = Photoelectrons(
            pixel=np.zeros(n_photoelectrons, dtype=np.int),
            time=np.zeros(n_photoelectrons),
            charge=np.ones(n_photoelectrons),
            metadata=dict(test="test"),
        )
        result = spectrum.apply(photoelectrons, rng)
        pe.append(result.get_photoelectrons_per_pixel(1)[0])
        charge.append(result.get_charge_per_pixel(1)[0])
    return np.array(pe), np.array(charge)


@pytest.mark.parametrize("spectrum_class", non_abstract_children(SPESpectrum))
def test_spe_spectra(spectrum_class):
    rng = np.random.RandomState(seed=3)

    spectrum = spectrum_class(normalise_charge=True)
    pe, charge = _get_result_photoelectrons(spectrum, rng)
    np.testing.assert_allclose(spectrum.average, 1, rtol=2e-2)
    np.testing.assert_allclose(pe.mean(), 1, rtol=2e-2)
    np.testing.assert_allclose(charge.mean(), 1, rtol=2e-2)
    # np.testing.assert_allclose(1+charge.std()**2, spectrum.excess_noise_factor, rtol=1e-2)

    spectrum = spectrum_class(normalise_charge=False)
    pe, charge = _get_result_photoelectrons(spectrum, rng)
    np.testing.assert_allclose(pe.mean(), 1, rtol=2e-2)
    np.testing.assert_allclose(charge.mean(), spectrum.average, rtol=2e-2)
    # np.testing.assert_allclose(1+charge.std()**2, spectrum.excess_noise_factor, rtol=1e-2)


def test_delayed():
    n_photoelectrons = 1000000
    photoelectrons = Photoelectrons(
        pixel=np.zeros(n_photoelectrons, dtype=np.int),
        time=np.full(n_photoelectrons, 10.),
        charge=np.ones(n_photoelectrons),
    )

    rng = np.random.RandomState(seed=1)

    spectrum_template = SiPMDelayed(spe_sigma=0.1, opct=0.2, time_constant=20)
    result = spectrum_template.apply(photoelectrons, rng)
    time = result.time[~result.initial]
    assert (time > 10).all()
    np.testing.assert_allclose(time.mean(), 10 + 20, rtol=1e-2)
    np.testing.assert_allclose(time.std(), 20, rtol=1e-2)
