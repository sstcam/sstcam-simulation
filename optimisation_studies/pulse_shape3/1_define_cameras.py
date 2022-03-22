import argparse
from os.path import join
import numpy as np
from sstcam_simulation import Camera, SSTCameraMapping
from sstcam_simulation.camera.spe import SiPMPrompt
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.camera.pulse import GaussianPulse, GenericPulse
from sstcam_simulation.camera.coupling import ACOffsetCoupling
from CHECLabPy.utils.files import create_directory
from matplotlib import pyplot as plt
from scipy import signal


def main():
    parser = argparse.ArgumentParser(description="Generate Camera configuration files")
    parser.add_argument('-o', dest='output', help='Output directory')
    args = parser.parse_args()
    output_dir = args.output

    create_directory(output_dir)

    n_samples = 96
    spe = SiPMPrompt(opct=0.1, normalise_charge=False)
    camera_kwargs = dict(
        mapping=SSTCameraMapping(),
        photoelectron_spectrum=spe,
        n_waveform_samples=n_samples,
        continuous_readout_duration=n_samples,
        readout_noise=GaussianNoise(stddev=2),
        digitisation_noise=GaussianNoise(stddev=1),
    )

    # Load chec-s pulse
    pulse_checs_t, pulse_checs_y = np.loadtxt("checs.dat", unpack=True)

    # Gaussian pulse
    width = 11
    sigma = width / 2.355
    pulse_gaussian = GaussianPulse(20, sigma, 40, mv_per_pe=1)

    # Leicester measurements
    pulse_lei00002_t, pulse_lei00002_y = np.loadtxt("C3--scope2--00002.txt", unpack=True)
    pulse_lei00007_t, pulse_lei00007_y = np.loadtxt("C3--scope2--00007.txt", unpack=True)
    pulse_lei00007lat_t, pulse_lei00007lat_y = np.loadtxt("C3--scope2--00007_lowamptail.txt", unpack=True)

    def plot(x, y, name):
        step = x[1] - x[0]

        peaks, _ = signal.find_peaks(y, height=0.5)
        widths, _, _, _ = signal.peak_widths(y, peaks, rel_height=0.5)
        fwhm = widths[0] * step
        _, _, rt_10, _ = signal.peak_widths(y, peaks, rel_height=0.9)
        _, _, rt_90, _ = signal.peak_widths(y, peaks, rel_height=0.1)
        rise_time = (rt_90[0] - rt_10[0]) * step

        label = f"{name}\n(FWHM={fwhm:.2f} ns, Tr={rise_time:.2f} ns)"
        plt.plot(x, y, label=label)

    plot(pulse_checs_t, pulse_checs_y, "CHEC-S")
    plot(pulse_lei00002_t, pulse_lei00002_y, "C3--scope2--00002")
    plot(pulse_lei00007_t, pulse_lei00007_y, "C3--scope2--00007")
    plot(pulse_lei00007lat_t, pulse_lei00007lat_y, "C3--scope2--00007_lowamptail")
    plot(pulse_gaussian.time, pulse_gaussian.amplitude, "Gaussian")
    plt.legend(loc="best")
    plt.xlabel("Time (ns)")
    plt.ylabel("Amplitude")
    plt.savefig("pulses.pdf")

    for mv_per_pe in [0.5, 1.0, 1.4, 2, 3.2, 4]:
        pulse = GenericPulse(pulse_checs_t, pulse_checs_y, mv_per_pe=mv_per_pe)
        coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
        camera = Camera(**camera_kwargs, photoelectron_pulse=pulse, coupling=coupling)
        name = f"checs_{mv_per_pe:.2f}.pkl"
        camera.attach_metadata("pulse_name", "CHEC-S")
        camera.save(join(output_dir, name))

        pulse = GaussianPulse(20, sigma, 40, mv_per_pe=mv_per_pe)
        coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
        camera = Camera(**camera_kwargs, photoelectron_pulse=pulse, coupling=coupling)
        name = f"gaussian_{mv_per_pe:.2f}.pkl"
        camera.attach_metadata("pulse_name", "Gaussian")
        camera.save(join(output_dir, name))

        pulse = GenericPulse(pulse_lei00002_t, pulse_lei00002_y, mv_per_pe=mv_per_pe)
        coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
        camera = Camera(**camera_kwargs, photoelectron_pulse=pulse, coupling=coupling)
        name = f"C3--scope2--00002_{mv_per_pe:.2f}.pkl"
        camera.attach_metadata("pulse_name", "C3--scope2--00002")
        camera.save(join(output_dir, name))

        pulse = GenericPulse(pulse_lei00007_t, pulse_lei00007_y, mv_per_pe=mv_per_pe)
        coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
        camera = Camera(**camera_kwargs, photoelectron_pulse=pulse, coupling=coupling)
        name = f"C3--scope2--00007_{mv_per_pe:.2f}.pkl"
        camera.attach_metadata("pulse_name", "C3--scope2--00007")
        camera.save(join(output_dir, name))

        pulse = GenericPulse(pulse_lei00007lat_t, pulse_lei00007lat_y, mv_per_pe=mv_per_pe)
        coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
        camera = Camera(**camera_kwargs, photoelectron_pulse=pulse, coupling=coupling)
        name = f"C3--scope2--00007_lowamptail_{mv_per_pe:.2f}.pkl"
        camera.attach_metadata("pulse_name", "C3--scope2--00007_lowamptail")
        camera.save(join(output_dir, name))


if __name__ == '__main__':
    main()
