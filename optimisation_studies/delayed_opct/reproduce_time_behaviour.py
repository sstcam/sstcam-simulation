from sstcam_simulation import Camera, PhotoelectronSource, EventAcquisition, SSTCameraMapping, Photoelectrons
from sstcam_simulation.camera.spe import SiPMDelayed, PerfectPhotosensor
import numpy as np
from scipy.optimize import curve_fit
import iminuit
from spefit.cost import _total_binned_nll, _sum_log_x, _bin_nll, _least_squares
from CHECLabPy.plotting.setup import Plotter
from matplotlib import pyplot as plt
from tqdm import trange
from IPython import embed


class TimeSeperationPlot(Plotter):
    pass


def main():
    camera = Camera(
        continuous_readout_duration=10000,
        mapping=SSTCameraMapping(n_pixels=1),
        photoelectron_spectrum=SiPMDelayed(opct=0.2, time_constant=24),
        # photoelectron_spectrum=PerfectPhotosensor(),
    )
    source = PhotoelectronSource(camera=camera)
    nsb_rate = 2.3
    n_events = 100000
    dt = []
    rng = np.random.RandomState(None)
    for iev in trange(n_events):
        pe = source.get_nsb(rate=nsb_rate)
        # pe = Photoelectrons(
        #     pixel=np.zeros(1),
        #     time=np.full(1, 20.),
        #     charge=np.ones(1),
        # )
        # pe = camera.photoelectron_spectrum.apply(pe, rng)
        time = np.sort(pe.time)
        # time = pe.time
        if time.size > 1:
            # embed()
            dt.append(time[1] - time[0])
            # dt.append(time[1:] - time[0])
            # if (np.diff(time) < 100).any():
            #     embed()
            # dt.append(np.diff(time))
    # dt = dt[dt < 1000]
    # dt = (time - time[:, None]).ravel()
    # dt = dt[dt > 8]
    # embed()
    dt = np.array(dt)
    # dt = np.concatenate(dt)
    dt = dt[(dt > 0) & (dt < 1000)]
    hist, edges = np.histogram(dt, bins=200)
    between = (edges[1:] + edges[:-1]) / 2


    def func(t, a0, tc0, a1, tc1):
        return a0 * 1/tc0 * np.exp(t/-tc0) + a1 * 1/tc1 * np.exp(t/-tc1)

    def cost_binned_nll(a0, tc0, a1, tc1):
        f_y = func(between, a0, tc0, a1, tc1)
        scale = np.sum(hist) / np.sum(f_y)
        return np.sum(_bin_nll(f_y * scale, hist))

    def cost_ls(a0, tc0, a1, tc1):
        f_y = func(between, a0, tc0, a1, tc1)
        gt5 = hist > 5
        return np.sum((hist[gt5] - f_y[gt5]) ** 2 / hist[gt5])

    def cost_unbinned_nll(a0, tc0, a1, tc1):
        if np.isnan(np.array([a0, tc0, a1, tc1])).any():
            return np.inf
        f_y = func(dt, a0, tc0, a1, tc1)
        return -_sum_log_x(f_y)

    initial = dict(
        a0=1,
        tc0=30,
        a1=1,
        tc1=1e3/nsb_rate
    )
    limits = dict(
        limit_a0=(0, None),
        limit_tc0=(1, None),
        limit_a1=(0, None),
        limit_tc1=(1, None)
    )
    fixed = dict(
        # fix_a0=True,
        # fix_tc0=True,
        # fix_a1=True,
        # fix_tc1=True
    )

    m0 = iminuit.Minuit(
        cost_binned_nll,
        **initial,
        **limits,
        **fixed,
        errordef=0.5,
        print_level=0,
        pedantic=False,
        throw_nan=True,
    )
    m0.migrad()
    m0.hesse()

    # # Attempt to run HESSE to compute parabolic errors.
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", iminuit.util.HesseFailedWarning)
    #     m0.hesse()

    # embed()
    print(m0.values)
    print(m0.errors)

    # popt, pcov = curve_fit(func, between, hist)
    # embed()
    p_timesep = TimeSeperationPlot(talk=True)
    # p_timesep.ax.hist(dt, bins=100)#, density=True)
    # p_timesep.ax.plot(between, hist, '.')
    (_, caps, _) = p_timesep.ax.errorbar(
        between, hist, yerr=np.sqrt(hist), mew=1, capsize=1, elinewidth=0.5,
        markersize=2, linewidth=0.5, fmt='.', zorder=1
    )
    # p_timesep.ax.errorbar(between, hist, yerr=np.sqrt(hist))
    f_y = func(between, *m0.values.values())
    scale = np.sum(hist) / np.sum(f_y)
    p_timesep.ax.plot(between, f_y * scale)
    p_timesep.ax.text(0.1, 0.8, fr"$τ_S$={m0.values['tc0']:.2f}", transform=p_timesep.ax.transAxes)
    p_timesep.ax.text(0.7, 0.4, fr"$τ_L$={m0.values['tc1']:.2f}", transform=p_timesep.ax.transAxes)
    p_timesep.ax.set_yscale("log")
    p_timesep.ax.set_xlabel("Δt (ns)")
    p_timesep.save("timesep.pdf")
    # p_timesep.ax.set_xscale("log")
    # plt.show()


if __name__ == '__main__':
    main()
