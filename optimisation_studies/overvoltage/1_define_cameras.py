import argparse
from os.path import join
import numpy as np
from sstcam_simulation import Camera, SSTCameraMapping
from sstcam_simulation.camera.spe import SiPMPrompt
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.camera.pulse import GaussianPulse
from sstcam_simulation.camera.coupling import ACOffsetCoupling
from CHECLabPy.utils.files import create_directory
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Generate Camera configuration files")
    parser.add_argument('-o', dest='output', help='Output directory')
    args = parser.parse_args()
    output_dir = args.output

    create_directory(output_dir)

    n_samples = 128
    width = 14
    sigma = width / 2.355

    for opct in np.linspace(0, 0.5, 6):
        for mv_per_pe in np.linspace(0.4, 4, 5):
            for nsb in np.linspace(0, 50, 5):
                pulse = GaussianPulse(20, sigma, 40, mv_per_pe=mv_per_pe)
                spe = SiPMPrompt(opct=opct, normalise_charge=False)
                coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
                camera = Camera(
                    mapping=SSTCameraMapping(),
                    photoelectron_spectrum=spe,
                    photoelectron_pulse=pulse,
                    coupling=coupling,
                    n_waveform_samples=n_samples,
                    continuous_readout_duration=n_samples,
                    readout_noise=GaussianNoise(stddev=0.5),
                    digitisation_noise=GaussianNoise(stddev=1),

                )
                camera.attach_metadata("nsb", nsb)
                name = f"camera_{opct:.2f}_{mv_per_pe:.2f}_{nsb:.2f}.pkl"
                camera.save(join(output_dir, name))


if __name__ == '__main__':
    main()
