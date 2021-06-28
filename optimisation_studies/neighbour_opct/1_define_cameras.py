import argparse
from os.path import join
from sstcam_simulation import Camera, SSTCameraMapping
from sstcam_simulation.camera.spe import SiPMReflectedOCT
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
    mapping = SSTCameraMapping()
    camera_kwargs = dict(
        mapping=mapping,
        photoelectron_pulse=pulse,
        n_waveform_samples=n_samples,
        continuous_readout_duration=n_samples,
        digitisation_noise=GaussianNoise(stddev=1),
    )

    self_opct_l = [0.08, 0.15, 0.3]
    reflected_opct_l = [0, 0.08, 0.15, 0.3]
    reflected_scale_l = [0.6, 1.0, 1.5, 2.3, 5]
    for self_opct in self_opct_l:
        for reflected_opct in reflected_opct_l:
            for reflected_scale in reflected_scale_l:
                spe = SiPMReflectedOCT(
                    opct=self_opct,
                    reflected_opct=reflected_opct,
                    reflected_scale=reflected_scale,
                    normalise_charge=False,
                    mapping=mapping,
                )
                coupling = ACOffsetCoupling(pulse_area=pulse.area, spectrum_average=spe.average)
                camera = Camera(**camera_kwargs, photoelectron_spectrum=spe, coupling=coupling)
                name = f"refl_opct_{self_opct:.2f}_{reflected_opct:.2f}_{reflected_scale:.2f}.pkl"
                camera.save(join(output_dir, name))


if __name__ == '__main__':
    main()
