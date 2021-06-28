import numpy as np
import pandas as pd
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.plotting.resolutions import ChargeResolutionPlotter
from scipy.interpolate import RegularGridInterpolator
from IPython import embed

# Requirements
CRREQ_2PE = ChargeResolutionPlotter.requirement(np.array([2]))[0]
CRREQ_20PE = ChargeResolutionPlotter.requirement(np.array([20]))[0]
CRREQ_200PE = ChargeResolutionPlotter.requirement(np.array([200]))[0]
CRREQ_2000PE = ChargeResolutionPlotter.requirement(np.array([2000]))[0]

CRREQ_2PE_50PDE = ChargeResolutionPlotter.requirement(np.array([2*0.5]))[0]
CRREQ_20PE_50PDE = ChargeResolutionPlotter.requirement(np.array([20*0.5]))[0]
CRREQ_200PE_50PDE = ChargeResolutionPlotter.requirement(np.array([200*0.5]))[0]
CRREQ_2000PE_50PDE = ChargeResolutionPlotter.requirement(np.array([2000*0.5]))[0]

MINIMGAMP_GAMMA_25PDE = 250 * 0.25
MINIMGAMP_PROTON_25PDE = 480 * 0.25
MINIMGAMP_GAMMA_50PDE = 250 * 0.5
MINIMGAMP_PROTON_50PDE = 480 * 0.5


class MIAContourPlot(Plotter):
    def __init__(self, interpolator, talk):
        self.interpolator = interpolator
        super().__init__(talk=talk)

    def plot_opct_vs_nsb(self, opct: np.ndarray, nsb: np.ndarray, mv_per_pe: float):
        xg, yg, zg = np.meshgrid(opct, nsb, mv_per_pe, indexing='ij')
        opct = xg.ravel()
        nsb = yg.ravel()
        mv_per_pe = zg.ravel()
        mia = self.interpolator(opct, nsb, mv_per_pe)

        c = self.ax.tricontourf(opct, nsb, mia, 15)
        self.ax.set_xlabel("OCT")
        self.ax.set_ylabel("NSB (MHz)")
        cb = self.fig.colorbar(c, ax=self.ax, label="Minimum Image Amplitude")

    def plot_mv_per_pe_vs_nsb(self, mv_per_pe: np.ndarray, nsb: np.ndarray, opct: float):
        xg, yg, zg = np.meshgrid(opct, nsb, mv_per_pe, indexing='ij')
        opct = xg.ravel()
        nsb = yg.ravel()
        mv_per_pe = zg.ravel()
        mia = self.interpolator(opct, nsb, mv_per_pe)

        c = self.ax.tricontourf(mv_per_pe, nsb, mia, 15)
        self.ax.set_xlabel("mV per p.e.")
        self.ax.set_ylabel("NSB (MHz)")
        cb = self.fig.colorbar(c, ax=self.ax, label="Minimum Image Amplitude")


class MinimumImageAmplitudeInterpolator:
    def __init__(self, df):
        xcol, ycol, zcol, vcol = "opct", "nsb_rate", "mv_per_pe", "minimum_image_amplitude"
        df = df.sort_values(by=[xcol, ycol, zcol])
        xvals = df[xcol].unique()
        yvals = df[ycol].unique()
        zvals = df[zcol].unique()
        vvals = df[vcol].values.reshape(len(xvals), len(yvals), len(zvals))

        self.f = RegularGridInterpolator((xvals, yvals, zvals), vvals)

    def __call__(self, opct, nsb, mv_per_pe):
        pts = np.column_stack([opct, nsb, mv_per_pe])
        return self.f(pts)


def main():
    with pd.HDFStore("performance.h5", mode='r') as store:
        df_ = store['data']
        df_g = df_.loc[df_['shower_primary_id'] == 0]
        df_p = df_.loc[df_['shower_primary_id'] == 101]

    talk = True

    interpolator = MinimumImageAmplitudeInterpolator(df_g)

    opct = np.linspace(0, 0.5, 100)
    nsb = np.linspace(0, 50, 100)
    mv_per_pe = np.linspace(0.4, 4, 100)

    p_2d = MIAContourPlot(interpolator, talk=talk)
    p_2d.plot_opct_vs_nsb(opct, nsb, 4)
    p_2d.save("mia_opct_vs_nsb.pdf")

    p_2d = MIAContourPlot(interpolator, talk=talk)
    p_2d.plot_mv_per_pe_vs_nsb(mv_per_pe, nsb, 0.08)
    p_2d.save("mia_mVperpe_vs_nsb.pdf")


if __name__ == '__main__':
    main()
