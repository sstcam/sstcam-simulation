from abc import ABCMeta, abstractmethod
import numpy as np
from numba import njit, vectorize, float64, int64

__all__ = [
    "single_gaussian",
    "sipm_gentile_spe",
    "SPESpectrum",
    "SiPMGentileSPE",
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


class SPESpectrum(metaclass=ABCMeta):
    def __init__(self, x_min, x_max, n_points):
        """
        Base for classes which define the probability density function of the
        charge for a single photoelectron.

        NOTE: this is **not** the spectrum for an "average illumination" i.e. a Poisson
        average close to one. This is the probability density function of the
        possible charges a single photoelectron can be measured as. This spectrum
        therefore defines the Excess Noise Factor (ENF) of the photosensor. Also,
        by definition, it does not include the pedestal peak.

        The spectrum is evaluated on initialisation.
        For a new spectrum to be defined, a new class should be initialised.

        Parameters
        ----------
        x_min : float
            Minimum charge at which the spectrum is defined (Unit: p.e.)
        x_max : float
            Maximum charge at which the spectrum is defined (Unit: p.e.)
        n_points: int
            Number of points between x_min and x_max used to define the spectrum
        """
        self.x_min = x_min
        self.x_max = x_max
        self.n_points = n_points

        # Calculate normalisation scale factors
        x = np.linspace(self.x_min, self.x_max, self.n_points)
        pdf = self._function(x)  # Evaluate at x
        x_scale = np.average(x, weights=pdf)
        pdf_scale = pdf.sum()

        self.x = x / x_scale
        self.pdf = pdf / pdf_scale

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
        variance = np.average((self.x - 1) ** 2, weights=self.pdf)
        return 1 + variance


class PerfectPhotosensor(SPESpectrum):
    def __init__(self):
        """
        SPE spectrum for a perfect photosensor, which always reports the exact
        number of photoelectrons that are generated inside it
        """
        super().__init__(x_min=1, x_max=1, n_points=1)

    def _function(self, x):
        return np.ones(1)


class SingleGaussianSPE(SPESpectrum):
    def __init__(self, x_min=0, x_max=10, n_points=10000, spe_sigma=0.1):
        self.spe_sigma = spe_sigma
        super().__init__(x_min=x_min, x_max=x_max, n_points=n_points)

    def _function(self, x):
        return single_gaussian(x, self.spe_sigma)


class SiPMGentileSPE(SPESpectrum):
    def __init__(self, x_min=0, x_max=10, n_points=10000, spe_sigma=0.1, opct=0.2):
        """
        SPE spectrum for an SiPM, using the Gentile formula

        Parameters
        ----------
        x_min : float
            Minimum charge at which the spectrum is defined (Unit: p.e.)
        x_max : float
            Maximum charge at which the spectrum is defined (Unit: p.e.)
        n_points: int
            Number of points between x_min and x_max used to define the spectrum
        spe_sigma : float
            Width of the single photoelectron peak
        opct : float
            Probability of optical crosstalk
        """
        self.spe_sigma = spe_sigma
        self.opct = opct
        super().__init__(x_min=x_min, x_max=x_max, n_points=n_points)

    def _function(self, x):
        return sipm_gentile_spe(x, self.spe_sigma, self.opct)
