import argparse
import numpy as np
from numba import njit, prange, vectorize, int64, float64
from math import lgamma, exp, pow, sqrt, log, pi
from functools import partial
from scipy.stats import poisson
from scipy.signal import convolve
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors
from tqdm import trange
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages


class PDFGenerator:

    def __init__(self, mn=0, mx=50, npoints=1000):
        self.x = np.linspace(mn, mx, npoints)
        self.prob = 0
        self.SQRT2PI = sqrt(2.0 * pi)

    def normal_pdf(self, mean=0, std_deviation=1):
        u = (self.x - mean) / std_deviation
        return np.exp(-0.5 * u ** 2) / (self.SQRT2PI * std_deviation)
        
    # SiPM Single Electron Spectrum (SES) - Mean = 1.0?
    def set_sipm(self, spe, spe_sigma, opct):
        pe_signal = 0
        # Loop over the possible total number of cells fired
        for k in range(1, 250):
            pk = (1-opct) * pow(opct, k-1)

            # Combine spread of pedestal and pe (and afterpulse) peaks
            pe_sigma = np.sqrt(k * spe_sigma ** 2)

            # Evaluate probability at each value of x
            pe_signal += pk * self.normal_pdf(k * spe, pe_sigma)

        self.prob = pe_signal / pe_signal.sum()

    def draw(self, out):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.plot(self.x, self.prob, color='dodgerblue')
        ax.set_xlabel("# pe")
        ax.set_yscale("log")
        ax.minorticks_on()
        ax.set_xlim(0, 10)
        ax.set_ylim(1e-8,self.prob.max()*1.2)
        out.savefig(fig)
        plt.close('all')


class PulseGenerator:

    def __init__(self):
        self.time = 0
        self.amplitude = 0

    # Simple pulse
    def set_gaussian(self, mean=0, std_deviation=1, samlen=0.25):
        t = np.arange(-20, 20, samlen)
        u = (t - mean) / std_deviation
        a = np.exp(-0.5 * u ** 2) / (sqrt(2.0 * pi)  * std_deviation)
        self.time = t
        self.amplitude = a / a.max()
    
    def draw(self, out):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.plot(self.time, self.amplitude, color='red')
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude (normalise)")
        ax.minorticks_on()
        out.savefig(fig)
        plt.close('all')


class WaveformGenerator:

    def __init__(self, pulse, pdf, wflen=128, samlen=0.25, t0 = 40, sigma_t = 0):
        self.pulse = pulse
        self.pdf = pdf
        self.wflen = wflen
        self.samlen = samlen
        self.t0 = t0
        self.sigma_t = sigma_t
        self.time = np.arange(0, wflen, samlen) # x-axis
        self.amplitude = np.zeros(len(self.time))
        self.cache_desc = []
        self.cache_time = []
        self.cache_amplitude = []
    
    def _add_nsb(self, mhz):
        # Number of NSB in this wf
        n_nsb = np.random.poisson(mhz * 1e6 * self.wflen * 1e-9)

        # Uniformly distribute NSB photons in time across wf
        nsb_times = np.random.choice(self.time, n_nsb) # Uniform if final p= parameter omitted

        # Fluctuate the amount of pe from each NSB photon
        nsb_pe = np.random.choice(self.pdf.x, (n_nsb), p=self.pdf.prob)

        # Work out the x bin for each NSB time
        nsb_pos = np.asarray(nsb_times * (1./self.samlen), dtype=np.int)

        # Set the number of pe in each of those bins
        self.amplitude[nsb_pos] += nsb_pe
    
    def _add_signal(self, n):
         # Choose a random number of photons
        n_ph = np.random.poisson(n)

        # Fluctuate to pe
        n_pe = np.random.choice(self.pdf.x, n_ph, p=self.pdf.prob).sum()

        # Add time jitter
        if self.sigma_t > 0:
            sig_pos = round(np.random.normal(self.t0, self.sigma_t) * (1./self.samlen))
        else:
            sig_pos = round(self.t0 * (1./self.samlen))

        # Add the pe to wf
        self.amplitude[sig_pos] += n_pe

    def _convolve_with_pulse(self):
        self.amplitude = convolve(self.amplitude, self.pulse.amplitude, mode='same')

    def _digitise(self, samrate_ghz):
        ratio = int(samrate_ghz/self.samlen) # per Ghz
        self.time = np.arange(0, self.wflen, 1./samrate_ghz)
        self.amplitude = np.mean(self.amplitude.reshape(-1, ratio), axis=1)
        # Add other digitisation effects??
    
    def _cache(self, desc='wf'):
        self.cache_desc.append(desc)
        self.cache_time.append(self.time)
        self.cache_amplitude.append(self.amplitude)

    def generate(self, nsb_MHz=60, signal=0, samrate_ghz=0, cache=False):

        # Clear the wf
        self.amplitude = np.zeros(len(self.time))
        
        # Add NSB
        if nsb_MHz > 0:
            self._add_nsb(nsb_MHz)
            if cache: self._cache('NSB positions')
        
        # Add the signal
        if signal > 0:
           self._add_signal(signal)
           if cache: self._cache('NSB & signal positions')

        # Convolve the NSB positions with a reference pulse
        self._convolve_with_pulse()
        if cache: self._cache('Convolved with pulse shape')
        
        # Add electronic noise
        # TODO

        # Digitise
        if samrate_ghz>0:
            self._digitise(samrate_ghz)
            if cache: self._cache('Digitised')
    
   
    def draw(self, out, thresh=None, trig_times=None):
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111)
        for d, x, y in zip(self.cache_desc, self.cache_time, self.cache_amplitude):
            ax.plot(x, y, drawstyle='steps-mid')
        
        if thresh is not None:
            ax.axhline(thresh, lw=0.75, color='black')

        if trig_times is not None:
            for t in trig_times:
                ax.axvline(t, lw=0.75, color='black')

        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Amplitude")
        out.savefig(fig)
        plt.close('all')


class TriggerGenerator:

    def threshold(self, wf, thresh, min_peak_width, min_peak_gap):
        peaks, _ = find_peaks(wf.amplitude, height=thresh, 
                              width=min_peak_width, distance=min_peak_gap)
        return len(peaks), peaks

    def scan_threshold(self, wf, thresh, min_peak_gap = 10, min_peak_width = 2):
        '''
        wf: Waveform to scan
        thresh: Array of thresholds to assses
        min_peak_gap: Minimum gap between peaks to trigger (ns)
        min_peak_width: Minimum width of peak to trigger (ns)
        '''
        ntrig = []
        for t in thresh:
            n, _ = self.threshold(wf, thresh=t, min_peak_width=min_peak_width, 
                                 min_peak_gap=min_peak_gap)
            ntrig.append(n)

        return np.asarray(ntrig)
        

def test_draw_wfs(nsb_MHz_perpix = 100, signal=0, wflen=128, thresh =0, fname_pdf='test_wfs.pdf'):
    
    out = PdfPages(fname_pdf)

    # Create a pulse
    samlen = 0.25 # 4 samples per ns
    pulse_sig = 10./2.63 # 10 ns FWHM
    pulse = PulseGenerator()
    pulse.set_gaussian(0, pulse_sig, samlen=samlen)
    pulse.draw(out)

    # Create a probability density function
    pdf = PDFGenerator()
    pdf.set_sipm(spe=1, spe_sigma=0.1, opct=0.2)
    pdf.draw(out)

    # WF
    wf = WaveformGenerator(pulse, pdf, wflen=wflen, samlen=samlen, t0=40, sigma_t=1)
    wf.generate(nsb_MHz=nsb_MHz_perpix * 4, signal=signal, samrate_ghz=1, cache=True)

    # Trigger
    if thresh > 0:
        trigger = TriggerGenerator()
        min_peak_gap = 10 / samlen # 10 ns # Minimum gap between peaks
        min_peak_width = 2 / samlen # Minimum peak width (time above thresh)
        ntrig, trig_times = trigger.threshold(wf, thresh=thresh, min_peak_width=min_peak_width, min_peak_gap=min_peak_gap)
        wf.draw(out, thresh, trig_times)
    else:
        wf.draw(out)
    out.close()



def generate_bias_curve(nsb_MHz_perpix=60, wflen=1e5, samlen=0.25, spe=1, spe_sigma=0.1, opct=0.2, sigma_t0=1, 
                        thresh_min=1, thresh_max=50, thresh_nsteps=50, n_ev=50, fname='bias'):
    
    # Create a pulse
    pulse = PulseGenerator()
    pulse.set_gaussian(0, 10./2.63)

    # Create a PDF
    pdf = PDFGenerator()
    pdf.set_sipm(spe, spe_sigma, opct)

    # WF
    wf = WaveformGenerator(pulse, pdf, wflen=wflen, samlen=samlen, t0=40, sigma_t=sigma_t0)

    # Trigger
    min_peak_gap = 10 / samlen # 10 ns # Minimum gap between peaks
    min_peak_width = 2 / samlen # Minimum peak width (time above thresh)
    trigger = TriggerGenerator()

    # Loop over threshold 
    thresh = np.linspace(thresh_min, thresh_max, thresh_nsteps)
    ntrig = np.zeros((n_ev, len(thresh)))
    for i in trange(n_ev):

        # Make wf for a SP 
        # No signal, no digitisation - trigger only!
        wf.generate(nsb_MHz=nsb_MHz_perpix * 4, signal=0, samrate_ghz=0, cache=False)

        # Determine number of peaks vs threshold
        ntrig[i] = trigger.scan_threshold(wf, thresh, min_peak_width=min_peak_width, min_peak_gap=min_peak_gap)
    
    rate = 1e6 * ntrig.mean(axis=0) / wflen #Hz
    rate_err = 1e6 * (ntrig.std(axis=0)/ sqrt(n_ev)) / wflen #Hz
    
    np.savez(fname, thresh, rate, rate_err)

    return thresh, rate, rate_err

def test_biascurves():
    
    out = PdfPages('biascurves.pdf')
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(111)
    
    nsb = [10, 40, 100, 200, 500]
    for n in nsb:
        t, r, r_err = generate_bias_curve(nsb_MHz_perpix=n)
        ax.errorbar(t, r, yerr=r_err)

    ax.axhline(30, color='grey')
    ax.set_yscale("log")
    ax.set_xlabel("Threshold (pe)")
    ax.set_ylabel("Rate (Hz)")
    ax.minorticks_on()
    out.savefig(fig)
    plt.close('all')
        
    out.close()

if __name__ == "__main__":

    description = 'Simple simulation of a single pixel trace'
    parser = argparse.ArgumentParser(description=description)
    #parser.add_argument('-i', '--input', type=str, required=True, help='DL1 or R1 file name, including path')
    #parser.add_argument('-o', '--out', type=str, help='Output file name, defaults to Run1234_plots.pdf/Run1234_plots-wf.pdf', default=None)
    #parser.add_argument('-n', '--maxevents', action='store', help='Number of events to process', type=int, default=1e5)

    args = parser.parse_args()

    test_draw_wfs(nsb_MHz_perpix=40, wflen=128, signal=50, thresh=0, fname_pdf='test_signal.pdf')
    test_draw_wfs(nsb_MHz_perpix=100, wflen=1000, signal=0, thresh=10, fname_pdf='test_nsb.pdf')
    test_biascurves()
    
    