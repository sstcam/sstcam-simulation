from abc import ABCMeta, abstractmethod
import numpy as np
from numba import njit, vectorize, float64, int64
from sstcam_simulation import Photoelectrons

__all__ = [
    "single_gaussian",
    "sipm_gentile_spe",
    "SPESpectrum",
    "SiPMGentileSPE",
    "SiPMPrompt",
    "SiPMDelayed"
]


SQRT2PI = np.sqrt(2.0 * np.pi)


@vectorize([float64(float64, float64, float64)], fastmath=True)
def normal_pdf(x, mean, std_deviation):
    """
    Probability density function of a normal distribution.
    Defined from first principles and vectorized so that it can be called from Numba

    Parameters
    ----------
    x : ndarray
        Charge (p.e.) to evaluate the pdf at
    mean : float
        Mean of the distribution
    std_deviation : float
        Standard deviation of the distribution
    Returns
    -------
    ndarray
    """
    u = (x - mean) / std_deviation
    return np.exp(-0.5 * u ** 2) / (SQRT2PI * std_deviation)


@njit(fastmath=True)
def single_gaussian(x, spe_sigma):
    """
    Simple photomultiplier tube SPE, defined as a normal distribution

    Parameters
    ----------
    x : ndarray
        Charge (p.e.) to evaluate the pdf at
    spe_sigma : float
        Width of the single photoelectron peak

    Returns
    -------
    ndarray
    """
    return normal_pdf(x, 1, spe_sigma)


@vectorize([float64(int64, float64)], fastmath=True)
def optical_crosstalk_probability(k, opct):
    """
    Calculate the probability for k microcells to be fired following the
    generation of 1 photoelectron in the SiPM pixel

    Parameters
    ----------
    k : int or ndarray
        Number of total cells fired
    opct : float or ndarray
        Probability that more than 1 microcell fires following the generation
        of 1 photoelectron in the SiPM

    Returns
    -------
    float
        Probability of k cells fired
    """
    if k <= 0:
        return 0
    return (1 - opct) * pow(opct, k - 1)


@njit(fastmath=True)
def sipm_gentile_spe(x, spe_sigma, opct):
    """
    SiPM SPE spectrum, modified from Gentile 2010
    http://adsabs.harvard.edu/abs/2010arXiv1006.3263G
    (modified to ignore afterpulsing contributions,  which are minimal for
    the SST camera silicon photomultipliers)

    Parameters
    ----------
    x : ndarray
        Charge (p.e.) to evaluate the pdf at
    spe_sigma : float
        Width of the single photoelectron peak
    opct : float
        Probability of optical crosstalk
    """
    pe_signal = np.zeros(x.size)
    # Loop over the possible total number of cells fired
    for k in range(1, 250):
        pk = optical_crosstalk_probability(k, opct)
        pe_sigma = np.sqrt(k * spe_sigma ** 2)
        pe_signal += pk * normal_pdf(x, k, pe_sigma)

    return pe_signal


def _generate_opct_pe(photoelectrons, rng, ipe, probability):
    n_initial = len(photoelectrons)
    n_repeat = rng.choice(ipe, p=probability, size=n_initial) - 1
    n_new = np.sum(n_repeat)

    # Repeat the existing photoelectrons to obtain the crosstalk photoelectrons
    time = np.repeat(photoelectrons.time, n_repeat)
    pixel = np.repeat(photoelectrons.pixel, n_repeat)
    charge = np.ones(n_new)
    initial = ~np.repeat(photoelectrons.initial, n_repeat)
    return Photoelectrons(pixel=pixel, time=time, charge=charge, initial=initial)


class SPESpectrum(metaclass=ABCMeta):
    def __init__(self, normalise_charge=True):
        """
        Base for classes which define the probability density function of the
        charge for a single photon detection.

        NOTE: this is **not** the spectrum for an "average illumination" i.e. a Poisson
        average close to one. This is the probability density function of the
        possible charges a single photoelectron can be measured as. This spectrum
        therefore defines the Excess Noise Factor (ENF) of the photosensor. Also,
        by definition, it does not include the pedestal peak.

        The spectrum is evaluated on initialisation.
        For a new spectrum to be defined, a new class should be initialised.

        Parameters
        ----------
        normalise_charge : bool
            If True, the charge is normalised to in units of photoelectrons. The
            average charge produced for a single photon detection will equal 1.
            If False, the charge in in units of number of fired microcells.
        """
        self.normalise_charge = normalise_charge

    @abstractmethod
    def apply(self, photoelectrons, rng):
        """
        Apply the spectrum to the photoelectrons

        Parameters
        ----------
        photoelectrons : Photoelectrons
            Container for the photoelectron arrays
        rng : RandomState
            Numpy RandomState for propagating seed

        Returns
        -------
        Photoelectrons
            Container for the photoelectron arrays with the spectrum applied
        """

    @property
    @abstractmethod
    def average(self):
        """
        Obtain the Excess Noise Factor (ENF) of the spectrum. This factor is
        commonly used to encompass the multiplicative errors in the
        amplification process of a photosensor, and directly informs about the
        charge resolution.

        Returns
        -------
        float
        """

    @property
    @abstractmethod
    def excess_noise_factor(self):
        """
        Obtain the Excess Noise Factor (ENF) of the spectrum. This factor is
        commonly used to encompass the multiplicative errors in the
        amplification process of a photosensor, and directly informs about the
        charge resolution.

        Returns
        -------
        float
        """


class PerfectPhotosensor(SPESpectrum):
    """
    SPE spectrum for a perfect photosensor, which always reports the exact
    number of photoelectrons
    """

    def apply(self, photoelectrons, rng):
        return Photoelectrons(
            pixel=photoelectrons.pixel,
            time=photoelectrons.time,
            charge=np.ones(len(photoelectrons)),
            metadata=photoelectrons.metadata
        )

    @property
    def average(self):
        return 1

    @property
    def excess_noise_factor(self):
        return 1


class SPESpectrumTemplate(SPESpectrum):
    def __init__(self, x_min=0, x_max=10, n_points=10000, normalise_charge=True):
        """
        Subset of SPESpectrum which define the spectrum in terms of a PDF template

        Parameters
        ----------
        x_min : float
            Minimum charge at which the spectrum is defined (Unit: p.e.)
        x_max : float
            Maximum charge at which the spectrum is defined (Unit: p.e.)
        n_points: int
            Number of points between x_min and x_max used to define the spectrum
        """
        super().__init__(normalise_charge=normalise_charge)
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points

        # Calculate normalisation scale factors
        self.x = np.linspace(self.x_min, self.x_max, self.n_points)
        self.pdf = self._function(self.x)  # Evaluate at x
        pdf_scale = self.pdf.sum()

        # Normalise X axis
        if normalise_charge:
            x_scale = np.average(self.x, weights=self.pdf)
            self.x /= x_scale

        # Normalise Y axis
        self.pdf /= pdf_scale

    def apply(self, photoelectrons, rng):
        # Inverse Transform Sampling
        charge = rng.choice(self.x, size=len(photoelectrons), p=self.pdf)
        return Photoelectrons(
            pixel=photoelectrons.pixel,
            time=photoelectrons.time,
            charge=charge,
            metadata=photoelectrons.metadata
        )

    @abstractmethod
    def _function(self, x):
        """
        Function which describes the spectrum

        Parameters
        ----------
        x : ndarray
            Charge (p.e.) to evaluate the pdf at

        Returns
        -------
        photoelectron_pulse : ndarray
            Probability for charge x (not-normalised)
        """

    @property
    def average(self):
        return np.average(self.x, weights=self.pdf)

    @property
    def excess_noise_factor(self):
        #TODO
        variance = np.average((self.x - self.average) ** 2, weights=self.pdf)
        return 1 + variance


class SiPMGentileSPE(SPESpectrumTemplate):
    def __init__(self, spe_sigma=0.1, opct=0.2, **kwargs):
        """
        SPE spectrum for an SiPM, using the Gentile formula

        Parameters
        ----------
        spe_sigma : float
            Width of the single photoelectron peak
        opct : float
            Probability of optical crosstalk
        kwargs
            Keyword arguments for SPESpectrum
        """
        self.spe_sigma = spe_sigma
        self.opct = opct
        super().__init__(**kwargs)

    def _function(self, x):
        return sipm_gentile_spe(x, self.spe_sigma, self.opct)


class SiPMPrompt(SPESpectrum):
    def __init__(self, spe_sigma=0.1, opct=0.2, **kwargs):
        """
        SPE spectrum for an SiPM, creating additional prompt photoelectrons

        Parameters
        ----------
        spe_sigma : float
            Width of the single photoelectron peak
        opct : float
            Probability of optical crosstalk
        kwargs
            Keyword arguments for SPESpectrum
        """
        self.spe_sigma = spe_sigma
        self.opct = opct
        self._ipe = np.arange(1, 250)
        self._p = optical_crosstalk_probability(self._ipe, opct)
        self._scale = 1/(1-self.opct)
        super().__init__(**kwargs)

    def apply(self, photoelectrons, rng):
        pe_opct = _generate_opct_pe(photoelectrons, rng, self._ipe, self._p)
        pe_total = photoelectrons + pe_opct

        # Fluctuate the charge
        pe_total.charge = rng.normal(1, self.spe_sigma, pe_total.charge.size)
        if self.normalise_charge:
            pe_total.charge /= self._scale

        return pe_total

    @property
    def average(self):
        return 1 if self.normalise_charge else self._scale

    @property
    def excess_noise_factor(self):
        #TODO
        return 1 + self.opct


class SiPMDelayed(SPESpectrum):
    def __init__(self, spe_sigma=0.1, opct=0.2, time_constant=19.8, **kwargs):
        """
        SPE spectrum for an SiPM, creating additional delayed photoelectrons

        Parameters
        ----------
        spe_sigma : float
            Width of the single photoelectron peak
        opct : float
            Probability of optical crosstalk
        kwargs
            Keyword arguments for SPESpectrum
        """
        self.spe_sigma = spe_sigma
        self.opct = opct
        self.time_constant = time_constant
        self._ipe = np.arange(1, 250)
        self._p = optical_crosstalk_probability(self._ipe, opct)
        self._scale = 1/(1-self.opct)
        super().__init__(**kwargs)

    def apply(self, photoelectrons, rng):
        pe_opct = _generate_opct_pe(photoelectrons, rng, self._ipe, self._p)
        pe_opct.time += rng.exponential(self.time_constant, size=pe_opct.time.size)
        pe_total = photoelectrons + pe_opct

        # Fluctuate the charge
        pe_total.charge = rng.normal(1, self.spe_sigma, pe_total.charge.size)
        if self.normalise_charge:
            pe_total.charge /= self._scale

        return pe_total

    @property
    def average(self):
        return 1 if self.normalise_charge else self._scale

    @property
    def excess_noise_factor(self):
        #TODO
        return 1 + self.opct
