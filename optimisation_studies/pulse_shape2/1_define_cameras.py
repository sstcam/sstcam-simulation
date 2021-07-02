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

    # for mv_per_pe in [0.5, 0.8, 1.1, 1.375, 2.25, 3.125, 4]:
    for mv_per_pe in [0.8, 1.1]:
        for width in [2, 4,  6, 8, 10, 14, 20]:
            sigma = width / 2.355
            pulse = GaussianPulse(30, sigma, 60, mv_per_pe=mv_per_pe)
            coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
            camera = Camera(**camera_kwargs, photoelectron_pulse=pulse, coupling=coupling)

            name = f"width_{width:.2f}_height_{mv_per_pe:.2f}.pkl"
            camera.save(join(output_dir, name))


if __name__ == '__main__':
    main()
