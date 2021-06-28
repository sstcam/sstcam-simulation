from sstcam_simulation import Photoelectrons, SSTCameraMapping
from sstcam_simulation.camera.spe import single_gaussian, sipm_gentile_spe, \
    SPESpectrum, optical_crosstalk_probability, SiPMDelayed, SiPMReflectedOCT
from ctapipe.core import non_abstract_children
import numpy as np
import pytest

subclasses = non_abstract_children(SPESpectrum)
subclasses.remove(SiPMReflectedOCT)


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


@pytest.mark.parametrize("opct", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_optical_crosstalk_probability_enf(opct):
    i = np.arange(1, 250)
    p = optical_crosstalk_probability(i, opct)
    avg = np.average(i, weights=p)
    var = np.average((i - avg) ** 2, weights=p)
    np.testing.assert_allclose(avg, 1/(1-opct))
    np.testing.assert_allclose(var, 1/(1/opct - 2 + opct))
    np.testing.assert_allclose(1 + var/avg**2, 1+opct)


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
            pixel=np.zeros(n_photoelectrons, dtype=int),
            time=np.zeros(n_photoelectrons),
            charge=np.ones(n_photoelectrons),
            metadata=dict(test="test"),
        )
        result = spectrum.apply(photoelectrons, rng)
        pe.append(result.get_photoelectrons_per_pixel(1)[0])
        charge.append(result.get_charge_per_pixel(1)[0])
    return np.array(pe), np.array(charge)


@pytest.mark.parametrize("spectrum_class", subclasses)
@pytest.mark.parametrize("normalise_charge", [True, False])
def test_spe_spectra(spectrum_class, normalise_charge):
    rng = np.random.RandomState(seed=3)

    spectrum = spectrum_class(normalise_charge=normalise_charge)
    pe, charge = _get_result_photoelectrons(spectrum, rng)
    np.testing.assert_allclose(pe.mean(), 1, rtol=2e-2)
    np.testing.assert_allclose(charge.mean(), spectrum.average, rtol=2e-2)
    enf = 1 + charge.std()**2 / charge.mean()**2
    np.testing.assert_allclose(enf, spectrum.excess_noise_factor, rtol=1e-1)

    if normalise_charge:
        np.testing.assert_allclose(spectrum.average, 1, rtol=2e-2)


def test_delayed():
    n_photoelectrons = 1000000
    photoelectrons = Photoelectrons(
        pixel=np.zeros(n_photoelectrons, dtype=int),
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


def test_neighbour():
    mapping = SSTCameraMapping(n_pixels=16)
    n_photoelectrons = 1000000
    photoelectrons = Photoelectrons(
        pixel=np.zeros(n_photoelectrons, dtype=int),
        time=np.full(n_photoelectrons, 10.),
        charge=np.ones(n_photoelectrons),
    )

    rng = np.random.RandomState(seed=1)

    spectrum_template = SiPMReflectedOCT(mapping=mapping, reflected_scale=5)
    result = spectrum_template.apply(photoelectrons, rng)
    charge = result.get_charge_per_pixel(mapping.n_pixels)
    assert (charge > 0).all()
