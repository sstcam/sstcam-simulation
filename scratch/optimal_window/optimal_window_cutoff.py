from sstcam_simulation.utils.efficiency import CameraEfficiency
from sstcam_simulation.utils.sipm import SiPMOvervoltage
from sstcam_simulation.utils.window_durham_needle import (
    Window,
    WindowDurhamNeedle,
    SSTWindowRun2,
    SSTWindowRun3,
    SSTWindowRun4,
    Prod4Window,
    DurhamNeedleWindowD2208Prod1FilterAR,
    AkiraWindow,
)
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from CHECLabPy.plotting.setup import Plotter

SIM_PATH = "/Users/Jason/Software/sstcam-simulation/optimisation_studies/overvoltage/performance.h5"
DATA_PATH = "cutoff.h5"

MINIMGAMP_GAMMA = 250
MINIMGAMP_PROTON = 480
BTEL1170_PDE = 0.2
BTEL0090_SNR = 0.35
PROD4_OPCT = 0.08
PROD4_MINIMGAMP_GAMMA = 153
PROD4_MINIMGAMP_PROTON = 208


class PEInterpolator:
    def __init__(self, df, column):
        xcol, ycol, zcol, vcol = "opct", "nsb_rate", "mv_per_pe", column
        df = df.sort_values(by=[xcol, ycol, zcol])
        xvals = df[xcol].unique()
        yvals = df[ycol].unique()
        zvals = df[zcol].unique()
        vvals = df[vcol].values.reshape(len(xvals), len(yvals), len(zvals))

        self.f = RegularGridInterpolator((xvals, yvals, zvals), vvals)

    def __call__(self, opct, nsb, mv_per_pe):
        pts = np.column_stack([opct, nsb, mv_per_pe])
        return self.f(pts)

    @classmethod
    def gamma(cls, column):
        with pd.HDFStore(SIM_PATH, mode='r') as store:
            df = store['data']
            df = df.loc[df['shower_primary_id'] == 0]
        return cls(df, column)

    @classmethod
    def proton(cls, column):
        with pd.HDFStore(SIM_PATH, mode='r') as store:
            df = store['data']
            df = df.loc[df['shower_primary_id'] == 101]
        return cls(df, column)


class WindowedWindow(Window):
    def __init__(self, rise_x, rise_y, fall_x):
        x = [0, rise_x-1, rise_x, fall_x-1, fall_x, 850, 1000]
        y = [0, 0,        rise_y, 0.9,     0.1,    0.1, 0.92]

        f = interp1d(x, y, fill_value='extrapolate')
        xnew = np.arange(200, 1000)
        ynew = f(xnew)
        super().__init__(np.array([0]), ynew[None, :])

    def weight_by_incidence_angle(self, off_axis_angle: float):
        return self.transmission[0]


def plot_comparison():
    fig, ax = plt.subplots()

    window = DurhamNeedleWindowD2208Prod1FilterAR()
    x = np.arange(200, 1000)
    y = window.interpolate_at_incidence_angle(45)
    ax.plot(x, y, label=f"Durham @45deg incidence")

    rise = 290
    window = WindowedWindow(280, 0.7, 520)
    y = window.transmission[0]
    ax.plot(x, y, label=f"Windowed-window @ {rise} nm")

    ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission")
    ax.set_ylim([0, 1])
    # ax.set_title(window.__class__.__name__)
    fig.savefig(f"window_comparison.pdf")


def generate_df():
    d_list = []

    sipm = SiPMOvervoltage.lvr3_6mm_50um_uncoated()
    sipm.overvoltage = 6
    pde_at_450nm = sipm.pde
    # mia_gamma_interp = PEInterpolator.gamma("minimum_image_amplitude")
    # trigger_threshold_mean_interp = PEInterpolator.gamma("trigger_threshold_mean")

    fall_x_array = np.linspace(300, 800, 100)

    for fall_x in tqdm(fall_x_array):
        window = WindowedWindow(280, 0.7, fall_x)
        eff = CameraEfficiency.from_sstcam(
            fov_angle=0,
            pde_at_450nm=pde_at_450nm,
            window=window,
        )

        opct = sipm.opct
        gain = sipm.gain
        camera_cherenkov_pde = eff.camera_cherenkov_pde
        mv_per_pe = 3.2
        nominal_nsb_rate = eff.nominal_nsb_rate.to_value("MHz")
        # mia_pe = mia_gamma_interp(opct, nominal_nsb_rate, mv_per_pe)[0]
        # mia_photons = mia_pe / camera_cherenkov_pde
        # trig_thresh_pe = trigger_threshold_mean_interp(opct, nominal_nsb_rate, mv_per_pe)[0]
        # trig_thresh_photons = trig_thresh_pe / camera_cherenkov_pde

        d_list.append(dict(
            fall_x=fall_x,
            pde_at_450nm=pde_at_450nm,
            opct=opct,
            gain=gain,
            B_TEL_1170_pde=eff.B_TEL_1170_pde,
            camera_cherenkov_pde=camera_cherenkov_pde,
            telescope_cherenkov_pde=eff.telescope_cherenkov_pde,
            camera_nsb_pde=eff.camera_nsb_pde.to_value(),
            telescope_nsb_pde=eff.telescope_nsb_pde.to_value(),
            camera_signal_to_noise=eff.camera_signal_to_noise.to_value(),
            telescope_signal_to_noise=eff.telescope_signal_to_noise.to_value(),
            nominal_nsb_rate=nominal_nsb_rate,
            maximum_nsb_rate=eff.maximum_nsb_rate.to_value("MHz"),
            n_cherenkov_photoelectrons=eff.n_cherenkov_photoelectrons,
            # minimum_image_amplitude=mia_photons,
            # trigger_threshold_pe=trig_thresh_pe,
            # trigger_threshold_photons=trig_thresh_photons,
        ))

    print(f"Writing {DATA_PATH}")
    df = pd.DataFrame(d_list)
    with pd.HDFStore(DATA_PATH) as store:
        store['data'] = df


def load_df():
    print(f"Reading {DATA_PATH}")
    with pd.HDFStore(DATA_PATH) as store:
        df = store['data']

    return df


def plot(df):
    print("Plotting results")
    prod4_eff = CameraEfficiency.from_prod4()

    p_snr = Plotter()
    x = df["fall_x"].values
    y = df["telescope_signal_to_noise"].values
    p_snr.ax.plot(x, y, '.', label="data")
    p_snr.ax.axhline(BTEL0090_SNR, ls='--', color='black', label="Requirement")
    p_snr.ax.axhline(prod4_eff.telescope_signal_to_noise, ls='--', color='blue', label="Prod4")
    p_snr.ax.set_xlabel("Cutoff (nm)")
    p_snr.ax.set_ylabel("B-TEL-0090 SNR")
    p_snr.add_legend(loc="best")
    p_snr.save("B-TEL-0090_snr.pdf")

    p_nominal_nsb = Plotter()
    x = df["fall_x"].values
    y = df["nominal_nsb_rate"].values
    p_nominal_nsb.ax.plot(x, y, '.', label="data")
    p_nominal_nsb.ax.axhline(prod4_eff.nominal_nsb_rate.to_value("MHz"), ls='--', color='blue', label="Prod4")
    p_nominal_nsb.ax.set_xlabel("Cutoff (nm)")
    p_nominal_nsb.ax.set_ylabel("Nominal NSB Rate (MHz)")
    p_nominal_nsb.add_legend(loc="best")
    p_nominal_nsb.save("nominal_nsb.pdf")


if __name__ == '__main__':
    plot_comparison()
    generate_df()
    df = load_df()
    plot(df)
