import numpy as np
import pandas as pd
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.plotting.resolutions import ChargeResolutionPlotter
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


class ComparisonPlot(Plotter):
    def plot(self, x, y, label):
        self.ax.plot(x, y, 'x', label=label)



def main():
    with pd.HDFStore("performance.h5", mode='r') as store:
        df_ = store['data']

        readout_noise = df_['readout_noise_stddev'].values
        digitisation_noise = df_['digitisation_noise_stddev'].values
        waveform_noise = np.sqrt((readout_noise / np.sqrt(5))**2 + digitisation_noise**2)

        df_['trigger_sn'] = df_['mv_per_pe'] / readout_noise
        df_['waveform_sn'] = df_['mv_per_pe'] / waveform_noise

        df_g = df_.loc[df_['shower_primary_id'] == 0]
        df_p = df_.loc[df_['shower_primary_id'] == 101]

    talk = True

    # embed()

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "minimum_image_amplitude"
    for trigger_sn, group in df_g.groupby("trigger_sn"):
        label = f"{trigger_sn:.2f}"
        p_comparison.plot(group['pulse_width'], group[performance_col], label)
    p_comparison.ax.axhline(MINIMGAMP_GAMMA_50PDE, ls='--', color='black')
    p_comparison.ax.text(3, MINIMGAMP_GAMMA_50PDE, "Requirement (Gamma, 50% PDE)", va='bottom', fontsize=5, color='black', alpha=0.4)
    p_comparison.ax.legend(title="Trigger S/N")
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Minimum Image Amplitude (p.e.)")
    p_comparison.save("minimum_image_amplitude_gamma.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "minimum_image_amplitude"
    for pulse_width, group in df_g.groupby("pulse_width"):
        label = f"{pulse_width:.2f}"
        p_comparison.plot(group['trigger_sn'], group[performance_col], label)
    p_comparison.ax.axhline(MINIMGAMP_GAMMA_50PDE, ls='--', color='black')
    p_comparison.ax.text(0.25, MINIMGAMP_GAMMA_50PDE, "Requirement (Gamma, 50% PDE)", va='bottom', fontsize=5, color='black', alpha=0.4)
    p_comparison.ax.legend(title="Pulse Width (ns)")
    p_comparison.ax.set_xlabel("Trigger S/N")
    p_comparison.ax.set_ylabel("Minimum Image Amplitude (p.e.)")
    p_comparison.save("minimum_image_amplitude_gamma_vs_sn.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "minimum_image_amplitude"
    for trigger_sn, group in df_p.groupby("trigger_sn"):
        label = f"{trigger_sn:.2f}"
        p_comparison.plot(group['pulse_width'], group[performance_col], label)
    p_comparison.ax.axhline(MINIMGAMP_PROTON_50PDE, ls='--', color='black')
    p_comparison.ax.text(3, MINIMGAMP_PROTON_50PDE, "Requirement (Proton, 50% PDE)", va='bottom', fontsize=5, color='black', alpha=0.4)
    p_comparison.ax.legend(title="Waveform S/N")
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Minimum Image Amplitude (p.e.)")
    p_comparison.save("minimum_image_amplitude_proton.pdf")
    #
    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_2pe"
    for waveform_sn, group in df_g.groupby("waveform_sn"):
        label = f"{waveform_sn:.2f}"
        p_comparison.plot(group['pulse_width'], group[performance_col], label)
    p_comparison.ax.axhline(CRREQ_2PE_50PDE, ls='--', color='black', label="Requirement (50% PDE)")
    p_comparison.ax.legend(title="Waveform S/N")
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Fractional CR @ 2 p.e.")
    p_comparison.save("cr_extracted_2pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_20pe"
    for waveform_sn, group in df_g.groupby("waveform_sn"):
        label = f"{waveform_sn:.2f}"
        p_comparison.plot(group['pulse_width'], group[performance_col], label)
    p_comparison.ax.axhline(CRREQ_20PE_50PDE, ls='--', color='black', label="Requirement (50% PDE)")
    p_comparison.ax.legend(title="Waveform S/N")
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Fractional CR @ 20 p.e.")
    p_comparison.save("cr_extracted_20pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_200pe"
    for waveform_sn, group in df_g.groupby("waveform_sn"):
        label = f"{waveform_sn:.2f}"
        p_comparison.plot(group['pulse_width'], group[performance_col], label)
    p_comparison.ax.axhline(CRREQ_200PE_50PDE, ls='--', color='black', label="Requirement (50% PDE)")
    p_comparison.ax.legend(title="Waveform S/N")
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Fractional CR @ 200 p.e.")
    p_comparison.save("cr_extracted_200pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_2000pe"
    for waveform_sn, group in df_g.groupby("waveform_sn"):
        label = f"{waveform_sn:.2f}"
        p_comparison.plot(group['pulse_width'], group[performance_col], label)
    p_comparison.ax.axhline(CRREQ_2000PE_50PDE, ls='--', color='black', label="Requirement (50% PDE)")
    p_comparison.ax.legend(title="Waveform S/N")
    p_comparison.ax.set_xlabel("Pulse Width (ns)")
    p_comparison.ax.set_ylabel("Fractional CR @ 2000 p.e.")
    p_comparison.save("cr_extracted_2000pe.pdf")


if __name__ == '__main__':
    main()
