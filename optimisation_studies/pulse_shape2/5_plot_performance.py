import numpy as np
import pandas as pd
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.plotting.resolutions import ChargeResolutionPlotter

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


class ComparisonPlot(Plotter):
    def plot(self, x, y, label):
        self.ax.plot(x, y, 'x', label=label)



def main():
    with pd.HDFStore("performance.h5", mode='r') as store:
        df_ = store['data']
        df_g = df_.loc[df_['shower_primary_id'] == 0]
        df_p = df_.loc[df_['shower_primary_id'] == 101]

    talk = True

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "minimum_image_amplitude"
    p_comparison.plot(df_g['pulse_width'], df_g[performance_col], None)
    p_comparison.ax.axhline(MINIMGAMP_GAMMA_50PDE, ls='--', color='black', label="Requirement (Gamma, 50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Minimum Image Amplitude (p.e.)")
    p_comparison.save("minimum_image_amplitude_gamma.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "minimum_image_amplitude"
    p_comparison.plot(df_p['pulse_width'], df_p[performance_col], None)
    p_comparison.ax.axhline(MINIMGAMP_PROTON_50PDE, ls='--', color='black', label="Requirement (Proton, 50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Minimum Image Amplitude (p.e.)")
    p_comparison.save("minimum_image_amplitude_proton.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_2pe"
    p_comparison.plot(df_g['pulse_width'], df_g[performance_col], "gamma")
    p_comparison.plot(df_p['pulse_width'], df_p[performance_col], "proton")
    p_comparison.ax.axhline(CRREQ_2PE_50PDE, ls='--', color='black', label="Requirement (50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Fractional CR @ 2 p.e.")
    p_comparison.save("cr_extracted_2pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_20pe"
    p_comparison.plot(df_g['pulse_width'], df_g[performance_col], "gamma")
    p_comparison.plot(df_p['pulse_width'], df_p[performance_col], "proton")
    p_comparison.ax.axhline(CRREQ_20PE_50PDE, ls='--', color='black', label="Requirement (50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Fractional CR @ 20 p.e.")
    p_comparison.save("cr_extracted_20pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_200pe"
    p_comparison.plot(df_g['pulse_width'], df_g[performance_col], "gamma")
    p_comparison.plot(df_p['pulse_width'], df_p[performance_col], "proton")
    p_comparison.ax.axhline(CRREQ_200PE_50PDE, ls='--', color='black', label="Requirement (50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Fractional CR @ 200 p.e.")
    p_comparison.save("cr_extracted_200pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_2000pe"
    p_comparison.plot(df_g['pulse_width'], df_g[performance_col], "gamma")
    p_comparison.plot(df_p['pulse_width'], df_p[performance_col], "proton")
    p_comparison.ax.axhline(CRREQ_2000PE_50PDE, ls='--', color='black', label="Requirement (50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Fractional CR @ 2000 p.e.")
    p_comparison.save("cr_extracted_2000pe.pdf")


if __name__ == '__main__':
    main()
