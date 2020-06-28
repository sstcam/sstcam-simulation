import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tables
from CHECLabPy.utils.resolutions import ChargeResolution


def extract_trigger_efficiency(illumination, n_triggers):
    df = pd.DataFrame(dict(illumination=illumination, n_triggers=n_triggers))
    trigger_eff = df.groupby('illumination').agg(['mean', 'std', 'count'])

    x = trigger_eff.index
    mean = trigger_eff['n_triggers']['mean'].values
    stddev = trigger_eff['n_triggers']['std'].values
    stderr = stddev / np.sqrt(trigger_eff['n_triggers']['count'].values)

    return x, mean, stderr


def bin_charge_resolution(df, n_bins=40):
    true = df['true'].values
    min_ = true.min()
    max_ = (true.max() // 500 + 1) * 500
    bins = np.geomspace(min_, max_, n_bins)
    bins = np.append(bins, 10**(np.log10(bins[-1]) + np.diff(np.log10(bins))[0]))
    df['bin'] = np.digitize(true, bins, right=True) - 1

    log = np.log10(bins)
    between = 10**((log[1:] + log[:-1]) / 2)
    edges = np.repeat(bins, 2)[1:-1].reshape((bins.size-1 , 2))
    edge_l = edges[:, 0]
    edge_r = edges[:, 1]
    df['between'] = between[df['bin']]
    df['edge_l'] = edge_l[df['bin']]
    df['edge_r'] = edge_r[df['bin']]

    return df


def extract_charge_resolution_mc(measured_charge, true_charge, n_bins=20):
    charge_resolution = ChargeResolution(mc_true=True)
    charge_resolution.add(0, true_charge.ravel(), measured_charge.ravel())
    res, _ = charge_resolution.finish()

    if n_bins:
        res = bin_charge_resolution(res, n_bins=n_bins)
        res = res.groupby('bin').mean()

    x = res['true'].values
    y = res['charge_resolution'].values

    return x, y


class TriggerEfficiencyPlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xscale('log')
        self.ax.set_xlabel("Illumination (p.e.)")
        self.ax.set_ylabel("Trigger Efficiency")
        self.ax.axhline(0.5, color='black', ls='-')

    def plot(self, x, y, yerr, teff_50, label=None):
        label += f" (50% @ {teff_50:.2f} p.e.)"
        color = self.ax._get_lines.get_next_color()
        self.ax.errorbar(x, y, yerr=yerr, fmt='.', color=color, label=label)
        self.ax.axvline(teff_50, ls=':', color=color)

    def save(self, path):
        self.ax.legend(loc='best')
        print(f"Saving plot: {path}")
        self.fig.savefig(path)


class ChargeResolutionPlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Number of Photoelectrons")
        self.ax.set_ylabel(r"Fractional Charge Resolution $\frac{{\sigma_Q}}{{Q}}$")
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

    def plot(self, x, y, label=None):
        self.ax.plot(x, y, '.', label=label)

    def save(self, path):
        self.ax.legend(loc='best')
        print(f"Saving plot: {path}")
        self.fig.savefig(path)


def main():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Plot the performance for a set of simulated "
                    "uniform lab illumination events",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', dest='input_path', help='path to the lab events hdf5 file'
    )
    args = parser.parse_args()

    input_path = args.input_path
    teff_path = input_path.replace("_lab.h5", "_teff.pdf")
    cr_path = input_path.replace("_lab.h5", "_cr.pdf")

    with tables.File(input_path, mode='r') as file:
        data = file.root.event[:]
        illumination = data['illumination']
        n_triggers = data['n_triggers']
        measured_charge = data['measured_charge']
        true_charge = data['true_charge']

    teff_x, teff_y, teff_yerr = extract_trigger_efficiency(illumination, n_triggers)
    teff_50 = np.interp(0.5, teff_y, teff_x)

    cr_x, cr_y = extract_charge_resolution_mc(measured_charge, true_charge)

    teff_plot = TriggerEfficiencyPlot()
    teff_plot.plot(teff_x, teff_y, teff_yerr, teff_50, label="Test")
    teff_plot.save(teff_path)

    cr_plot = ChargeResolutionPlot()
    cr_plot.plot(cr_x, cr_y, label="Test")
    cr_plot.save(cr_path)


if __name__ == '__main__':
    main()
