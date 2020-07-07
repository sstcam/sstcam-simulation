from measure_pulse import extract_width, extract_area
import argparse
from os.path import exists
import re
import tables
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
from tqdm import tqdm
from sstcam_simulation import Camera
from sstcam_simulation.performance.plot_lab_performance import \
    extract_charge_resolution_mc, extract_trigger_efficiency


def obtain_df_files(h5_paths):
    df_files = []
    pattern = r".*\/undershoot_(.+)_(.+)_(.+)_lab.h5"
    for h5_path in h5_paths:
        camera_path = h5_path.replace("_lab.h5", ".pkl")
        regexr = re.search(pattern, h5_path)
        ratio = float(regexr.group(1))
        sigma0 = float(regexr.group(2))
        sigma1 = float(regexr.group(3))

        assert exists(h5_path)
        assert exists(camera_path)

        camera = Camera.load(camera_path)
        x = camera.reference_pulse.time
        y = camera.reference_pulse.pulse

        pulse_width, undershoot_width = extract_width(x, y)
        if sigma1 == 0:
            undershoot_width = 0

        area_pos, area_neg = extract_area(x, y)

        df_files.append(dict(
            h5_path=h5_path,
            camera_path=camera_path,
            ratio=ratio,
            sigma0=sigma0,
            sigma1=sigma1,
            pulse_width=pulse_width,
            undershoot_width=undershoot_width,
            measured_ratio=area_neg/area_pos,
        ))
    return pd.DataFrame(df_files).sort_values(['sigma0', 'sigma1'])


def obtain_df_performance(df_files):
    df_list = []
    for _, row in tqdm(df_files.iterrows(), total=df_files.index.size):
        with tables.File(row['h5_path'], mode='r') as file:
            data = file.root.event[:]
            illumination = data['illumination']
            n_triggers = data['n_triggers']
            measured_charge = data['measured_charge']
            true_charge = data['true_charge']

        # Calibrate charge
        mask = (true_charge >= 50) & (true_charge <= 500)
        coeff = polyfit(true_charge[mask], measured_charge[mask], [1])
        measured_charge /= coeff[1]

        teff_x, teff_y, teff_yerr = extract_trigger_efficiency(illumination, n_triggers)
        teff_50 = np.interp(0.5, teff_y, teff_x)

        cr_x, cr_y = extract_charge_resolution_mc(measured_charge, true_charge)
        cr_2pe = np.interp(2, cr_x, cr_y)
        cr_100pe = np.interp(100, cr_x, cr_y)

        df_list.append(dict(
            ratio=row['ratio'],
            sigma0=row['sigma0'],
            sigma1=row['sigma1'],
            pulse_width=row['pulse_width'],
            undershoot_width=row['undershoot_width'],
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

    df_files = obtain_df_files(h5_paths)
    df = obtain_df_performance(df_files)
    with pd.HDFStore("performance.h5") as store:
        store['data'] = df


if __name__ == '__main__':
    main()
