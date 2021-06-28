import argparse
from os.path import join
import numpy as np
from sstcam_simulation import Camera, SSTCameraMapping
from sstcam_simulation.camera.spe import SiPMPrompt
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation.camera.coupling import ACOffsetCoupling
from CHECLabPy.utils.files import create_directory


def main():
    parser = argparse.ArgumentParser(description="Generate Camera configuration files")
    parser.add_argument('-o', dest='output', help='Output directory')
    args = parser.parse_args()
    output_dir = args.output

    create_directory(output_dir)

    n_samples = 128
    spe = SiPMPrompt(opct=0.1, normalise_charge=False)
    camera_kwargs = dict(
        mapping=SSTCameraMapping(),
        photoelectron_spectrum=spe,
        n_waveform_samples=n_samples,
        continuous_readout_duration=n_samples,
        readout_noise=GaussianNoise(stddev=0.15),
        digitisation_noise=GaussianNoise(stddev=1),
    )

    widths = np.linspace(2, 20, 10)
    for width in widths:
        sigma = width / 2.355
        pulse = GaussianPulse(30, sigma, 60, mv_per_pe=4)
        coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
        camera = Camera(**camera_kwargs, photoelectron_pulse=pulse, coupling=coupling)

        name = f"width_{width:.2f}.pkl"
        camera.save(join(output_dir, name))


if __name__ == '__main__':
    main()
