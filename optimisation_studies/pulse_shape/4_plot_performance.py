import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from CHECLabPy.utils.files import create_directory
from CHECLabPy.plotting.resolutions import ChargeResolutionPlotter

# Requirements
CRREQ_2PE = ChargeResolutionPlotter.requirement(np.array([2]))[0]
CRREQ_100PE = ChargeResolutionPlotter.requirement(np.array([100]))[0]

# CHEC-S (from data)
CHECS_RATIO = 0.20581828565154767
CHECS_PULSE_WIDTH = 10.650883938789999
CHECS_UNDERSHOOT_WIDTH = 19.880996171267505


def regular_grid_interp(df, n_undershoot, n_pulse):
    xi = np.linspace(df['undershoot_width'].min(), df['undershoot_width'].max(), n_undershoot)
    yi = np.linspace(df['pulse_width'].min(), df['pulse_width'].max(), n_pulse)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi_teff_50 = griddata((df['undershoot_width'], df['pulse_width']), df['teff_50'], (Xi, Yi))
    Zi_cr_2pe = griddata((df['undershoot_width'], df['pulse_width']), df['cr_2pe'], (Xi, Yi))
    Zi_cr_100pe = griddata((df['undershoot_width'], df['pulse_width']), df['cr_100pe'], (Xi, Yi))
    df_interp = pd.DataFrame(dict(
        undershoot_width=Xi.ravel(),
        pulse_width=Yi.ravel(),
        teff_50=Zi_teff_50.ravel(),
        cr_2pe=Zi_cr_2pe.ravel(),
        cr_100pe=Zi_cr_100pe.ravel(),
    ))
    return df_interp.dropna()


def plot_tricontourf(df, output_path):
    fig = plt.figure(figsize=(10, 15))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    def plot(ax, z, z_label, req=None):
        c = ax.tricontourf(df['undershoot_width'], df['pulse_width'], df[z], 15)
        ax.plot(df['undershoot_width'], df['pulse_width'], '.', c='black')
        ax.plot(CHECS_UNDERSHOOT_WIDTH, CHECS_PULSE_WIDTH, 'x', c='red', ms=15)
        ax.text(CHECS_UNDERSHOOT_WIDTH, CHECS_PULSE_WIDTH, "    (CHEC-S)", color='red')
        ax.set_xlabel("Undershoot Width (ns)")
        ax.set_ylabel("Pulse Width (ns)")
        ax.set_xlim(0, 25)
        ax.set_ylim(ymax=16)
        cb = fig.colorbar(c, ax=ax, label=z_label)
        cb.ax.plot([0, 10], [req, req], 'w-')

    plot(ax1, 'teff_50', "Amplitude @ 50% Trigger Efficiency (p.e)")
    plot(ax2, 'cr_2pe', "Fractional Charge Resolution @ 2 p.e.", CRREQ_2PE)
    plot(ax3, 'cr_100pe', "Fractional Charge Resolution @ 100 p.e.", CRREQ_100PE)

    print(f"Creating figure: {output_path}")
    fig.savefig(output_path, dpi=300)


def plot_width_ratio(df, output_path):
    df_interp = regular_grid_interp(df, 20, 20)
    df_interp['width_ratio'] = df_interp['undershoot_width'] / df_interp['pulse_width']

    fig = plt.figure(figsize=(10, 15))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    def plot(ax, y, y_label, req=None):
        #ax.hist2d(df_interp['width_ratio'], df_interp[y], bins=(40, 40))
        ax.plot(df_interp['width_ratio'], df_interp[y], '.')
        ax.axvline(CHECS_UNDERSHOOT_WIDTH/CHECS_PULSE_WIDTH, ls='--', color='r', alpha=0.5, label="CHEC-S")
        if req:
            ax.axhline(req, ls='--', color='black', alpha=0.5, label="Requirement")
        ax.set_xlabel("Undershoot Width / Pulse Width")
        ax.set_ylabel(y_label)
        #fig.colorbar(c, ax=ax, label="N")
        ax.legend(loc=1)

    plot(ax1, 'teff_50', "Amplitude @ 50% Trigger Efficiency (p.e)")
    plot(ax2, 'cr_2pe', "Fractional Charge Resolution @ 2 p.e.", CRREQ_2PE)
    plot(ax3, 'cr_100pe', "Fractional Charge Resolution @ 100 p.e.", CRREQ_100PE)

    print(f"Creating figure: {output_path}")
    fig.savefig(output_path, dpi=300)


def plot_1d_pulse_width(df, output_path):
    df_interp = regular_grid_interp(df, 7, 20)

    fig = plt.figure(figsize=(10, 15))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    def plot(ax, y, y_label, req=None):
        for undershoot_width, group in df_interp.groupby('undershoot_width'):
            ax.plot(group['pulse_width'], group[y], label=f"UW = {undershoot_width:.2f} ns")

        ax.axvline(CHECS_PULSE_WIDTH, ls='--', color='r', alpha=0.5, label="CHEC-S")
        if req:
            ax.axhline(req, ls='--', color='black', alpha=0.5, label="Requirement")

        ax.set_xlabel("Pulse Width (ns)")
        ax.set_ylabel(y_label)
        ax.legend(loc=2)

    plot(ax1, 'teff_50', "Amplitude @ 50% Trigger Efficiency (p.e)")
    plot(ax2, 'cr_2pe', "Fractional Charge Resolution @ 2 p.e.", CRREQ_2PE)
    plot(ax3, 'cr_100pe', "Fractional Charge Resolution @ 100 p.e.", CRREQ_100PE)

    print(f"Creating figure: {output_path}")
    fig.savefig(output_path, dpi=300)


def plot_1d_undershoot_width(df, output_path):
    df_interp = regular_grid_interp(df, 20, 10)

    fig = plt.figure(figsize=(10, 15))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    def plot(ax, y, y_label, req=None):
        for pulse_width, group in df_interp.groupby('pulse_width'):
            ax.plot(group['undershoot_width'], group[y], label=f"PW = {pulse_width:.2f} ns")

        ax.axvline(CHECS_UNDERSHOOT_WIDTH, ls='--', color='r', alpha=0.5, label="CHEC-S")
        if req:
            ax.axhline(req, ls='--', color='black', alpha=0.5, label="Requirement")

        ax.set_xlabel("Undershoot Width (ns)")
        ax.set_ylabel(y_label)
        ax.legend(loc='best')

    plot(ax1, 'teff_50', "Amplitude @ 50% Trigger Efficiency (p.e)")
    plot(ax2, 'cr_2pe', "Fractional Charge Resolution @ 2 p.e.", CRREQ_2PE)
    plot(ax3, 'cr_100pe', "Fractional Charge Resolution @ 100 p.e.", CRREQ_100PE)

    print(f"Creating figure: {output_path}")
    fig.savefig(output_path, dpi=300)


def plot_best(df, output_path):
    def get_df_best(dfi, perf):
        df_interp = regular_grid_interp(dfi, 20, 20)
        df_best = df_interp.loc[df_interp.groupby('pulse_width')[perf].idxmin()]
        return df_best

    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(3, 2, 1)
    ax2 = fig.add_subplot(3, 2, 2)
    ax3 = fig.add_subplot(3, 2, 3)
    ax4 = fig.add_subplot(3, 2, 4)
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)

    def plot_perf(ax, y, y_label, req=None):
        for ratio, group in df.groupby('ratio'):
            ratiop = int(ratio*100)
            df_best = get_df_best(group, y)
            ax.plot(df_best['pulse_width'], df_best[y], label=f"{ratiop}% Undershoot")

        ax.axvline(CHECS_PULSE_WIDTH, ls='--', color='r', alpha=0.5, label="CHEC-S")
        if req:
            ax.axhline(req, ls='--', color='black', alpha=0.5, label="Requirement")

        ax.set_xlabel("Pulse Width (ns)")
        ax.set_ylabel(y_label)
        ax.legend(loc=1)

    def plot_uw(ax, y):
        for ratio, group in df.groupby('ratio'):
            ratiop = int(ratio*100)
            df_best = get_df_best(group, y)
            ax.plot(df_best['pulse_width'], df_best['undershoot_width'], label=f"{ratiop}% Undershoot")

        ax.plot(CHECS_PULSE_WIDTH, CHECS_UNDERSHOOT_WIDTH, 'x', color='red', ms=15)
        ax.text(CHECS_PULSE_WIDTH, CHECS_UNDERSHOOT_WIDTH, "    (CHEC-S)", color='red')

        ax.set_xlabel("Pulse Width (ns)")
        ax.set_ylabel("Corresponding Undershoot Width (ns)")
        ax.legend(loc=1)

    plot_perf(ax1, 'teff_50', "BEST Amplitude @ 50% Trigger Efficiency (p.e)")
    plot_uw(ax2, 'teff_50')
    plot_perf(ax3, 'cr_2pe', "BEST Fractional Charge Resolution @ 2 p.e.", CRREQ_2PE)
    plot_uw(ax4, 'cr_2pe')
    plot_perf(ax5, 'cr_100pe', "BEST Fractional Charge Resolution @ 100 p.e.", CRREQ_100PE)
    plot_uw(ax6, 'cr_100pe')

    print(f"Creating figure: {output_path}")
    fig.savefig(output_path, dpi=300)


def main():
    with pd.HDFStore("performance.h5", mode='r') as store:
        df_data = store['data']

    # Select fully sampled region
    df_data = df_data.loc[
        (df_data['pulse_width'] <= 20) &
        (df_data['undershoot_width'] <= 30)
    ]

    # 20% UNDERSHOOT
    df = df_data.loc[df_data['ratio'] == 0.2]

    create_directory("figures")
    plot_tricontourf(df, "figures/tricontourf.png")
    plot_width_ratio(df, "figures/widthratio.pdf")
    plot_1d_pulse_width(df, "figures/1d_pulse_width.pdf")
    plot_1d_undershoot_width(df, "figures/1d_undershoot_width.pdf")
    plot_best(df_data, "figures/best.pdf")


if __name__ == '__main__':
    main()
