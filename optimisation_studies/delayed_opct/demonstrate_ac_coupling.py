from sstcam_simulation import Camera, PhotoelectronSource, EventAcquisition, SSTCameraMapping, Photoelectrons
from sstcam_simulation.camera.spe import SiPMDelayed, PerfectPhotosensor, SiPMPrompt
from sstcam_simulation.camera.coupling import NoCoupling, ACOffsetCoupling, ACFilterCoupling
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


def main():
    nsb_rate = 100
    spectrum = SiPMPrompt(opct=0.2, normalise_charge=False)
    pulse = GaussianPulse(mv_per_pe=4)

    coupling = {
        "No Coupling": NoCoupling(),
        "Filter AC Coupling": ACFilterCoupling(),
        'Offset AC Coupling': ACOffsetCoupling(nsb_rate, pulse.area, spectrum.average),
    }

    p_readout = WaveformPlot(talk=True)

    for name, c in coupling.items():
        camera = Camera(
            continuous_readout_duration=int(6e5),
            # continuous_readout_duration=int(200),
            mapping=SSTCameraMapping(n_pixels=1),
            photoelectron_spectrum=spectrum,
            photoelectron_pulse=pulse,
            coupling=c,
        )
        source = PhotoelectronSource(camera=camera, seed=1)
        acquisition = EventAcquisition(camera=camera)
        pe = source.get_nsb(nsb_rate)
        readout = acquisition.get_continuous_readout(pe)[0]
        # p_readout.ax.plot(camera.continuous_readout_time_axis, readout)
        label = name + f" (Avg = {readout.mean():.2f})"
        p_readout.ax.plot(camera.continuous_readout_time_axis[:1000], readout[:1000], label=label)
        # p_readout.ax.plot(camera.continuous_readout_time_axis[-1000:], readout[-1000:], label=label)

    p_readout.add_legend('best')
    p_readout.ax.set_xlabel("Time (ns)")
    p_readout.ax.set_ylabel("Amplitude (mV)")
    p_readout.ax.set_title(f"NSB Rate = {nsb_rate:.2f} MHz")
    p_readout.save("coupling.pdf")

if __name__ == '__main__':
    main()
