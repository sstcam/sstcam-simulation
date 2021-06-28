import argparse
import numpy as np
import pandas as pd
from sstcam_simulation.performance.plot_lab_performance import \
    bin_charge_resolution, ChargeResolution
from matplotlib import pyplot as plt
from CHECLabPy.plotting.resolutions import ChargeResolutionPlotter
from tqdm import tqdm
from collections import defaultdict


class CRGatherer:
    def __init__(self):
        self.charge_resolution = ChargeResolution(mc_true=True)

    def add(self, df):
        self.charge_resolution._df_list.append(df)

    def finish(self):
        res, _ = self.charge_resolution.finish()
        res = bin_charge_resolution(res, n_bins=20)
        res = res.groupby('bin').mean()
        x = res['true'].values
        y = res['charge_resolution'].values
        return x, y


class TriggerGatherer:
    def __init__(self):
        self.triggers = []

    def add(self, df):
        self.triggers.append(df)

    def finish(self):
        df_triggers = pd.concat(self.triggers, ignore_index=True)
        df_ia = df_triggers.groupby('image_amplitude')['triggered'].agg(['sum', 'count']).reset_index()
        df_binned = bin_charge_resolution(df_ia, column='image_amplitude')
        df_binned = df_binned.groupby('bin').mean()
        image_amplitude = df_binned['image_amplitude'].values
        trigger_success = df_binned['sum'] / df_binned['count']
        return image_amplitude, trigger_success


class CRPlotter:
    def __init__(self):
        self.plot = ChargeResolutionPlotter()

    def add(self, x, y, label):
        self.plot._plot(x, y, None, label=label)

    def save(self, path):
        x = np.geomspace(0.1, 1000, 100)
        self.plot.plot_poisson(x)
        self.plot.plot_requirement(x)
        self.plot.ax.set_xlabel("Illumination (p.e.)")
        self.plot.ax.set_ylabel("Fractional Charge Resolution")
        self.plot.add_legend(loc='best')
        self.plot.save(path)


class TriggerPlotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def add(self, x, y):
        self.ax.plot(x, y)

    def save(self, path):
        print("Figure saved to: ", path)
        self.ax.set_xscale('log')
        self.ax.set_xlabel("Image Amplitude (p.e.)")
        self.ax.set_ylabel("Trigger Eff.")
        self.fig.savefig(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='performance_paths', nargs='+',
                        help="paths to performance output files")
    args = parser.parse_args()
    paths = args.performance_paths

    n_total_events = defaultdict(int)
    cr_extracted_collection = defaultdict(CRGatherer)
    cr_signal_collection = defaultdict(CRGatherer)
    triggers_collection = defaultdict(TriggerGatherer)
    trigger_threshold_collection = defaultdict(list)
    pedestal_collection = defaultdict(list)
    for path in tqdm(paths):
        with pd.HDFStore(path) as store:
            metadata = store['metadata'].iloc[0].to_dict()
            shower_primary_id = metadata['shower_primary_id']
            pulse_width = metadata['pulse_width']
            mv_per_pe = metadata['mv_per_pe']
            readout_noise_stddev = metadata['readout_noise_stddev']
            digitisation_noise_stddev = metadata['digitisation_noise_stddev']
            trigger_threshold = metadata['trigger_threshold']
            pedestal = metadata['pedestal']

            key = (shower_primary_id, pulse_width, mv_per_pe, readout_noise_stddev, digitisation_noise_stddev)

            trigger_threshold_collection[key].append(trigger_threshold)
            pedestal_collection[key].append(pedestal)

            cr_extracted_collection[key].add(store['cr_extracted'])
            cr_signal_collection[key].add(store['cr_signal'])

            df_triggers = store['triggers']
            triggers_collection[key].add(df_triggers)
            n_total_events[key] += df_triggers.index.size

    cr_extracted_plot = CRPlotter()
    cr_signal_plot = CRPlotter()
    trigger_plot = TriggerPlotter()

    keys = cr_extracted_collection.keys()
    d_list = []
    for key in tqdm(keys):
        shower_primary_id, pulse_width, mv_per_pe, readout_noise_stddev, digitisation_noise_stddev = key

        cr_extracted_x, cr_extracted_y = cr_extracted_collection[key].finish()
        cr_extracted_2pe = np.interp(2, cr_extracted_x, cr_extracted_y)
        cr_extracted_20pe = np.interp(20, cr_extracted_x, cr_extracted_y)
        cr_extracted_200pe = np.interp(200, cr_extracted_x, cr_extracted_y)
        cr_extracted_2000pe = np.interp(2000, cr_extracted_x, cr_extracted_y)
        cr_extracted_plot.add(cr_extracted_x, cr_extracted_y, None)

        cr_signal_x, cr_signal_y = cr_signal_collection[key].finish()
        cr_signal_2pe = np.interp(2, cr_signal_x, cr_signal_y)
        cr_signal_20pe = np.interp(20, cr_signal_x, cr_signal_y)
        cr_signal_200pe = np.interp(200, cr_signal_x, cr_signal_y)
        cr_signal_2000pe = np.interp(2000, cr_signal_x, cr_signal_y)
        cr_signal_plot.add(cr_signal_x, cr_signal_y, None)

        image_amplitude, trigger_success = triggers_collection[key].finish()
        minimum_image_amplitude = np.interp(0.5, trigger_success, image_amplitude)
        trigger_plot.add(image_amplitude, trigger_success)

        trigger_threshold_mean = np.mean(trigger_threshold_collection[key])
        trigger_threshold_std = np.std(trigger_threshold_collection[key])

        pedestal_mean = np.mean(pedestal_collection[key])
        pedestal_std = np.std(pedestal_collection[key])

        d_list.append(dict(
            n_total_events=n_total_events[key],
            n_runs=len(trigger_threshold_collection),
            shower_primary_id=shower_primary_id,
            pulse_width=pulse_width,
            mv_per_pe=mv_per_pe,
            readout_noise_stddev=readout_noise_stddev,
            digitisation_noise_stddev=digitisation_noise_stddev,
            trigger_threshold_mean=trigger_threshold_mean,
            trigger_threshold_std=trigger_threshold_std,
            pedestal_mean=pedestal_mean,
            pedestal_std=pedestal_std,
            cr_extracted_2pe=cr_extracted_2pe,
            cr_extracted_20pe=cr_extracted_20pe,
            cr_extracted_200pe=cr_extracted_200pe,
            cr_extracted_2000pe=cr_extracted_2000pe,
            cr_signal_2pe=cr_signal_2pe,
            cr_signal_20pe=cr_signal_20pe,
            cr_signal_200pe=cr_signal_200pe,
            cr_signal_2000pe=cr_signal_2000pe,
            minimum_image_amplitude=minimum_image_amplitude,
        ))

    cr_extracted_plot.save("cr_extracted.pdf")
    cr_signal_plot.save("cr_signal.pdf")
    trigger_plot.save("trigger.pdf")

    df_performance = pd.DataFrame(d_list)
    output_path = "performance.h5"
    print("Creating file: ", output_path)
    with pd.HDFStore(output_path) as store:
        store['data'] = df_performance


if __name__ == '__main__':
    main()
