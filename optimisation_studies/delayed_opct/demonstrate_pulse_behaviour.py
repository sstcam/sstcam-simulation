from sstcam_simulation import Camera, PhotoelectronSource, EventAcquisition, SSTCameraMapping, Photoelectrons
from sstcam_simulation.camera.spe import SiPMDelayed, PerfectPhotosensor, SiPMPrompt
from sstcam_simulation.camera.pulse import GenericPulse
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


def main():
    normalise_charge = True
    spectra = {
        "No OCT": SiPMPrompt(opct=0, normalise_charge=normalise_charge),
        "Prompt OCT (20%)": SiPMPrompt(opct=0.2, normalise_charge=normalise_charge),
        "Delayed OCT (20%, τ=24ns)": SiPMDelayed(opct=0.2, time_constant=24, normalise_charge=normalise_charge),
        "Delayed OCT (55%, τ=24ns)": SiPMDelayed(opct=0.55, time_constant=24, normalise_charge=normalise_charge),
    }

    path = "/Users/Jason/Software/TargetCalib/source/dev/reference_pulse_checs_V1-1-0.cfg"
    ref_x, ref_y = np.loadtxt(path, unpack=True)
    pulse = GenericPulse(ref_x*1e9, ref_y, mv_per_pe=4)

    p_waveform = WaveformPlot(talk=True)

    for name, spectrum in spectra.items():
        camera = Camera(
            continuous_readout_duration=128,
            n_waveform_samples=128,
            mapping=SSTCameraMapping(n_pixels=1),
            photoelectron_spectrum=spectrum,
            photoelectron_pulse=pulse,
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
    p_waveform.ax.set_ylabel("Amplitude (mV)")
    p_waveform.ax.set_title("Average Pulse (50 p.e.)")
    p_waveform.save("pulse.pdf")

if __name__ == '__main__':
    main()
