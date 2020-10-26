from landau import nb_landau
from measure_pulse import extract_area, extract_width
import argparse
from os.path import join
import numpy as np
from numba import njit
from tqdm import tqdm
from sstcam_simulation import Camera
from sstcam_simulation.camera.spe import SiPMGentileSPE
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.camera.pulse import GenericPulse
from CHECLabPy.utils.files import create_directory


@njit
def pulse_(x, scale, sigma0, sigma1):
    pulse0_pos = 35
    for mpv1 in np.arange(pulse0_pos+5, 120, 0.1):
        y0 = nb_landau(x, pulse0_pos, sigma0)
        y0 /= y0.sum() * (x[1] - x[0])
        y0 *= scale

        if sigma1 > 0:
            y1 = nb_landau(x, mpv1, sigma1)
            y1 /= y1.sum() * (x[1] - x[0])
            y1 *= -1
        else:
            y1 = np.zeros(y0.size)

        y = y0 + y1
        y /= np.abs(y.sum() * (x[1] - x[0]))

        start_positive = (y[:y.argmax()] >= -1e-10).all()

        if start_positive:
            return y


def main():
    parser = argparse.ArgumentParser(description="Generate Camera configuration files")
    parser.add_argument('-o', dest='output', help='Output directory')
    args = parser.parse_args()
    output_dir = args.output

    create_directory(output_dir)

    # Define camera (CHEC-S)
    n_samples = 128
    camera_kwargs = dict(
        photoelectron_spectrum=SiPMGentileSPE(x_max=20, spe_sigma=0.12, opct=0.4),
        n_waveform_samples=n_samples,
        continuous_readout_duration=n_samples,
        readout_noise=GaussianNoise(stddev=0.15),
    )

    x = np.linspace(0, 128, 1000)

    n_sigma0 = 35
    n_sigma1 = 40
    ratio_values = [0.2, 0.4, 0.6, 0.8]
    sigma0_values = np.linspace(0.5, 20, n_sigma0)
    sigma1_values = np.linspace(0, 20, n_sigma1)

    pulse_width = np.zeros((n_sigma0, n_sigma1))
    undershoot_width = np.zeros((n_sigma0, n_sigma1))

    for ratio in tqdm(ratio_values):
        for isigma0, sigma0 in tqdm(enumerate(sigma0_values), total=n_sigma0):
            for isigma1, sigma1 in tqdm(enumerate(sigma1_values), total=n_sigma1):
                for scale in np.arange(0.1, 100, 0.04):  # Find scale required for ratio
                    y = pulse_(x, scale, sigma0, sigma1)

                    area_pos, area_neg = extract_area(x, y)
                    if np.sqrt((area_neg/area_pos - ratio)**2) < 0.05:
                        break
                    if sigma1 == 0:
                        scale = 1
                        break
                else:
                    continue

                y = pulse_(x, scale, sigma0, sigma1)
                width_pos, width_neg = extract_width(x, y)
                pulse_width[isigma0, isigma1] = width_pos
                undershoot_width[isigma0, isigma1] = width_neg if sigma1 > 0 else 0

                pulse = GenericPulse(x, y)
                camera = Camera(**camera_kwargs, photoelectron_pulse=pulse)
                name = f"undershoot_{ratio:.2f}_{sigma0:.2f}_{sigma1:.2f}.pkl"
                camera.save(join(output_dir, name))


if __name__ == '__main__':
    main()
