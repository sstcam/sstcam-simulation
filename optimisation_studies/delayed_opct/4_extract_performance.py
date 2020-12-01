import argparse
import tables
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit
from sstcam_simulation.performance.plot_lab_performance import \
    extract_charge_resolution_mc, extract_trigger_efficiency
from matplotlib import pyplot as plt
from CHECLabPy.plotting.resolutions import ChargeResolutionPlotter
from tqdm import tqdm
from IPython import embed


def plot_trigger_efficiency(x, y, yerr, teff_50, path):
    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt='.')
    ax.axhline(0.5, color='black', ls='--')
    ax.axvline(teff_50, color='black', ls=':')
    ax.set_xscale('log')
    ax.set_xlabel("Illumination (p.e.)")
    ax.set_ylabel("Trigger Efficiency")
    print(f"Saving plot: {path}")
    fig.savefig(path)


def plot_charge_resolution(x, y, path):
    plot = ChargeResolutionPlotter()
    plot._plot(x, y, None, label="simulation")
    plot.plot_poisson(x)
    plot.plot_requirement(x)
    plot.ax.set_xlabel("Illumination (p.e.)")
    plot.ax.set_ylabel("Fractional Charge Resolution")
    plot.add_legend(loc='best')
    plot.save(path)


def obtain_df_performance(h5_paths):
    df_list = []
    for h5_path in tqdm(h5_paths):
        with tables.File(h5_path, mode='r') as file:
            data = file.root.event[:]
            illumination = data['illumination']
            n_triggers = data['n_triggers']
            measured_charge = data['measured_charge']
            true_charge = data['true_charge']

            metadata = file.root.event.attrs
            input_opct = metadata['input_opct']
            measured_opct = metadata['measured_opct']
            measured_opct_err = metadata['measured_opct_err']
            time_constant = metadata['time_constant']

        # Calibrate charge
        # mask = (true_charge >= 50) & (true_charge <= 500)
        # coeff = polyfit(true_charge[mask], measured_charge[mask], [1])
        # measured_charge /= coeff[1]
        measured_charge *= (1 - measured_opct) / (1 - input_opct)

        teff_path = h5_path.replace(".h5", "_teff.pdf")
        cr_path = h5_path.replace(".h5", "_cr.pdf")

        teff_x, teff_y, teff_yerr = extract_trigger_efficiency(illumination, n_triggers)
        teff_50 = np.interp(0.5, teff_y, teff_x)

        cr_x, cr_y = extract_charge_resolution_mc(measured_charge, true_charge)
        cr_2pe = np.interp(2, cr_x, cr_y)
        cr_100pe = np.interp(100, cr_x, cr_y)

        # plot_trigger_efficiency(teff_x, teff_y, teff_yerr, teff_50, teff_path)
        # plot_charge_resolution(cr_x, cr_y, cr_path)

        df_list.append(dict(
            input_opct=input_opct,
            measured_opct=measured_opct,
            measured_opct_err=measured_opct_err,
            time_constant=time_constant,
            teff_50=teff_50,
            cr_2pe=cr_2pe,
            cr_100pe=cr_100pe,
        ))
    return pd.DataFrame(df_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='h5_paths', nargs='+',
                        help="paths to run files from extract_lab_events")
    args = parser.parse_args()
    h5_paths = args.h5_paths

    df = obtain_df_performance(h5_paths)

    path = "performance.h5"
    print("Creating file: ", path)
    with pd.HDFStore(path) as store:
        store['data'] = df


if __name__ == '__main__':
    main()
