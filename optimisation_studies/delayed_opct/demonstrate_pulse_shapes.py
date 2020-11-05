from sstcam_simulation import Camera, PhotoelectronSource, EventAcquisition, SSTCameraMapping, Photoelectrons
from sstcam_simulation.camera.spe import SiPMDelayed, PerfectPhotosensor, SiPMPrompt
from sstcam_simulation.camera.pulse import GaussianPulse
import numpy as np
from scipy.optimize import curve_fit
import iminuit
from spefit.cost import _total_binned_nll, _sum_log_x, _bin_nll, _least_squares
from CHECLabPy.plotting.setup import Plotter
from matplotlib import pyplot as plt
from tqdm import trange
from IPython import embed


class WaveformPlot(Plotter):
    pass


def process(height, normalise_charge, ylabel, output_path):
    spectra = {
        "Width=4, OCT=0%": SiPMPrompt(opct=0, normalise_charge=normalise_charge),
        "Width=4, OCT=20%": SiPMPrompt(opct=0.2, normalise_charge=normalise_charge),
        "Width=6, OCT=0%": SiPMPrompt(opct=0, normalise_charge=normalise_charge),
        "Width=6, OCT=20%": SiPMPrompt(opct=0.2, normalise_charge=normalise_charge),
    }
    pulse = {
        "Width=4, OCT=0%": GaussianPulse(sigma=4, mv_per_pe=height, mean=20, duration=40),
        "Width=4, OCT=20%": GaussianPulse(sigma=4, mv_per_pe=height, mean=20, duration=40),
        "Width=6, OCT=0%": GaussianPulse(sigma=6, mv_per_pe=height, mean=20, duration=40),
        "Width=6, OCT=20%": GaussianPulse(sigma=6, mv_per_pe=height, mean=20, duration=40),
    }
    p_waveform = WaveformPlot(talk=True)
    for name in spectra.keys():
        camera = Camera(
            continuous_readout_duration=128,
            n_waveform_samples=128,
            mapping=SSTCameraMapping(n_pixels=1),
            photoelectron_spectrum=spectra[name],
            photoelectron_pulse=pulse[name],
        )

        source = PhotoelectronSource(camera=camera)
        acquisition = EventAcquisition(camera=camera)
        n_events = 100
        waveform = np.zeros((n_events, 128))
        for iev in trange(n_events):
            pe = source.get_uniform_illumination(30, 50)
            readout = acquisition.get_continuous_readout(pe)
            waveform[iev] = acquisition.get_sampled_waveform(readout)[0]
        waveform_avg = np.mean(waveform, 0)

        p_waveform.ax.plot(waveform_avg, label=name)

    p_waveform.add_legend('best')
    p_waveform.ax.set_xlabel("Time (ns)")
    p_waveform.ax.set_ylabel(ylabel)
    # p_waveform.ax.set_title("Average Pulse (50 p.e.)")
    p_waveform.save(output_path)


def main():
    process(None, True, "Amplitude (p.e./sample)", "pulse_normalisation/default.pdf")
    process(1, True, "Amplitude (p.e.)", "pulse_normalisation/height.pdf")
    process(None, False, "Amplitude (f.c./sample)", "pulse_normalisation/spectra.pdf")
    process(1, False, "Amplitude (mV)", "pulse_normalisation/both.pdf")


if __name__ == '__main__':
    main()
