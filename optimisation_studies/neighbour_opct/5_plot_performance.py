import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.plotting.resolutions import ChargeResolutionPlotter

# Requirements
CRREQ_2PE = ChargeResolutionPlotter.requirement(np.array([2]))[0]
CRREQ_20PE = ChargeResolutionPlotter.requirement(np.array([20]))[0]
CRREQ_200PE = ChargeResolutionPlotter.requirement(np.array([200]))[0]
CRREQ_2000PE = ChargeResolutionPlotter.requirement(np.array([2000]))[0]

CRREQ_2PE_50PDE = ChargeResolutionPlotter.requirement(np.array([2/2]))[0]
CRREQ_20PE_50PDE = ChargeResolutionPlotter.requirement(np.array([20/2]))[0]
CRREQ_200PE_50PDE = ChargeResolutionPlotter.requirement(np.array([200/2]))[0]
CRREQ_2000PE_50PDE = ChargeResolutionPlotter.requirement(np.array([2000/2]))[0]

MINIMGAMP_GAMMA = 125
MINIMGAMP_PROTON = 480


class ComparisonPlot(Plotter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.colors = {
            0.08: 'green',
            0.15: 'orange',
            0.30: 'red'
        }
        self.markers = {
            0.00: "$1$",
            0.08: "$2$",
            0.15: "$3$",
            0.30: "$4$",
        }
        self.linestyles = {
            0.00: "-",
            0.08: "--",
            0.15: "-.",
            0.30: ":",
        }

    def plot(self, peformance, self_opct, reflected_opct, reflected_scale):
        color = self.colors[self_opct]
        marker = self.markers[reflected_opct]
        linestyle = self.linestyles[reflected_opct]

        self.ax.plot(reflected_scale, peformance, marker=marker, linestyle=linestyle, color=color)

    def add_requirement(self, y, label):
        self.ax.axhline(y, color='black', ls='--', alpha=0.4)
        self.ax.text(1, y, label, va='bottom', fontsize=5, color='black', alpha=0.4)

    def add_legend(self):
        legend_elements = []
        for self_opct, color in self.colors.items():
            legend_elements.append(Line2D([0], [0], marker='o', color=color, ls=' ', label=self_opct))

        legend = self.ax.legend(handles=legend_elements, loc='upper right', title="Self OCT", bbox_to_anchor=(1.35, 1))
        self.ax.add_artist(legend)

        legend_elements = []
        for reflected_opct in self.markers.keys():
            marker = self.markers[reflected_opct]
            linestyle = self.linestyles[reflected_opct]
            legend_elements.append(Line2D([0], [0], marker=marker, color='black', ls=linestyle, label=reflected_opct))

        self.ax.legend(handles=legend_elements, loc='lower right', title="Reflected OCT", bbox_to_anchor=(1.35, 0))



def main():
    with pd.HDFStore("performance.h5", mode='r') as store:
        df = store['data']
        df = df.loc[df['shower_primary_id'] == 0]

    talk = True

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "minimum_image_amplitude"
    for key, group in df.groupby(["self_opct", "reflected_opct"]):
        self_opct, reflected_opct = key
        p_comparison.plot(group[performance_col], self_opct, reflected_opct, group['reflected_scale'])
    p_comparison.add_requirement(MINIMGAMP_GAMMA, "Requirement (Gamma)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Reflected Scale (∝ Window Distance)")
    p_comparison.ax.set_ylabel("Minimum Image Amplitude (p.e.)")
    p_comparison.save("minimum_image_amplitude.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_2pe"
    for key, group in df.groupby(["self_opct", "reflected_opct"]):
        self_opct, reflected_opct = key
        p_comparison.plot(group[performance_col], self_opct, reflected_opct, group['reflected_scale'])
    p_comparison.add_requirement(CRREQ_2PE, "Requirement (25% PDE)")
    # p_comparison.add_requirement(CRREQ_2PE_50PDE, "Requirement (50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Reflected Scale (∝ Window Distance)")
    p_comparison.ax.set_ylabel("Fractional CR @ 2 p.e.")
    # p_comparison.ax.set_yscale('log')
    p_comparison.save("cr_extracted_2pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_20pe"
    for key, group in df.groupby(["self_opct", "reflected_opct"]):
        self_opct, reflected_opct = key
        p_comparison.plot(group[performance_col], self_opct, reflected_opct, group['reflected_scale'])
    p_comparison.add_requirement(CRREQ_20PE, "Requirement (25% PDE)")
    # p_comparison.add_requirement(CRREQ_20PE_50PDE, "Requirement (50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Reflected Scale (∝ Window Distance)")
    p_comparison.ax.set_ylabel("Fractional CR @ 20 p.e.")
    # p_comparison.ax.set_yscale('log')
    p_comparison.save("cr_extracted_20pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_200pe"
    for key, group in df.groupby(["self_opct", "reflected_opct"]):
        self_opct, reflected_opct = key
        p_comparison.plot(group[performance_col], self_opct, reflected_opct, group['reflected_scale'])
    p_comparison.add_requirement(CRREQ_200PE, "Requirement (25% PDE)")
    # p_comparison.add_requirement(CRREQ_200PE_50PDE, "Requirement (50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Reflected Scale (∝ Window Distance)")
    p_comparison.ax.set_ylabel("Fractional CR @ 200 p.e.")
    # p_comparison.ax.set_yscale('log')
    p_comparison.save("cr_extracted_200pe.pdf")

    p_comparison = ComparisonPlot(talk=talk)
    performance_col = "cr_extracted_2000pe"
    for key, group in df.groupby(["self_opct", "reflected_opct"]):
        self_opct, reflected_opct = key
        p_comparison.plot(group[performance_col], self_opct, reflected_opct, group['reflected_scale'])
    p_comparison.add_requirement(CRREQ_2000PE, "Requirement (25% PDE)")
    # p_comparison.add_requirement(CRREQ_2000PE_50PDE, "Requirement (50% PDE)")
    p_comparison.add_legend()
    p_comparison.ax.set_xlabel("Reflected Scale (∝ Window Distance)")
    p_comparison.ax.set_ylabel("Fractional CR @ 2000 p.e.")
    # p_comparison.ax.set_yscale('log')
    p_comparison.save("cr_extracted_2000pe.pdf")


if __name__ == '__main__':
    main()
