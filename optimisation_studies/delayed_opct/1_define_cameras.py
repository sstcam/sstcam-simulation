import argparse
from os.path import join
from sstcam_simulation import Camera
from sstcam_simulation.camera.spe import SiPMDelayed
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation.camera.coupling import ACOffsetCoupling
from CHECLabPy.utils.files import create_directory
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate Camera configuration files")
    parser.add_argument('-o', dest='output', help='Output directory')
    args = parser.parse_args()
    output_dir = args.output

    create_directory(output_dir)

    # Define camera
    n_samples = 128
    width = 8  # ns
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    pulse = GaussianPulse(mean=15, sigma=sigma, duration=30, mv_per_pe=4)
    camera_kwargs = dict(
        photoelectron_pulse=pulse,
        n_waveform_samples=n_samples,
        continuous_readout_duration=n_samples,
        digitisation_noise=GaussianNoise(stddev=1),
    )

    # from scipy.signal import find_peaks, peak_widths
    #
    # def _extract_widths(pulse_y):
    #     peaks, _ = find_peaks(pulse_y)
    #     return peak_widths(pulse_y, peaks)
    #
    # def extract_width(pulse_x, pulse_y):
    #     sample_width = pulse_x[1] - pulse_x[0]
    #     pulse_width = _extract_widths(pulse_y)[0][0] * sample_width
    #
    #     undershoot_widths = _extract_widths(-pulse_y)
    #     if len(undershoot_widths[0]) == 0 :
    #         undershoot_width = 0
    #     else:
    #         undershoot_width = undershoot_widths[0][-1] * sample_width
    #     return pulse_width, undershoot_width
    #
    # from matplotlib import pyplot as plt
    # plt.plot(pulse.time, pulse.amplitude)
    # plt.show()
    # width_pos, width_neg = extract_width(pulse.time, pulse.amplitude)

    # print(width_pos, width_neg)

    time_constant_list = [0, 5, 10, 20, 50, 100, 200, 1000]
    opct_list = np.linspace(0.01, 0.99, 50)
    for time_constant in time_constant_list:
        for opct in opct_list:
            spe = SiPMDelayed(opct=opct, time_constant=time_constant, normalise_charge=False)
            coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
            camera = Camera(**camera_kwargs, photoelectron_spectrum=spe, coupling=coupling)
            name = f"opct{opct:.1f}_tc{time_constant:.0f}.pkl"
            camera.save(join(output_dir, name))


if __name__ == '__main__':
    main()
