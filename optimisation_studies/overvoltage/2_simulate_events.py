from sstcam_simulation import Camera
from sstcam_simulation.performance import ChargeExtractor, CherenkovShowerGenerator, \
    EventsWriter, obtain_trigger_threshold, obtain_pedestal
import argparse
from tqdm import tqdm


def main():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Run a camera definition file through the sstcam-simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', dest='pe_path', help='Path to the photoelectrons file'
    )
    parser.add_argument(
        '-c', dest='camera_path', help='Path to the camera definition file'
    )
    parser.add_argument(
        '-o', dest='output_path', help='Path to the camera definition file'
    )
    parser.add_argument(
        '--trigger', default=600, type=float, dest='trigger_rate',
        help='Desired camera trigger rate'
    )
    args = parser.parse_args()

    pe_path = args.pe_path
    camera_definition_path = args.camera_path
    output_path = args.output_path
    trigger_rate = args.trigger_rate

    bias_scan_path = output_path.replace(".h5", "_biasscan.pdf")

    print(f"Loading camera: {camera_definition_path}")
    camera = Camera.load(camera_definition_path)
    nsb_rate = camera.nsb

    # Determine calibration
    extractor = ChargeExtractor.from_camera(camera)
    pedestal = obtain_pedestal(camera, extractor, nsb_rate)
    # 1.5x CR camera trigger rate (~600 Hz) == 2x dark NSB camera trigger rate
    threshold = obtain_trigger_threshold(camera, nsb_rate*2, trigger_rate, bias_scan_path)

    camera.update_trigger_threshold(threshold)
    camera.coupling.update_nsb_rate(nsb_rate)

    generator = CherenkovShowerGenerator(
        path=pe_path,
        camera=camera,
        extractor=extractor,
        pedestal=pedestal,
        nsb_rate=nsb_rate
    )
    with EventsWriter(output_path, generator.event_table_layout) as writer:
        writer.add_metadata(
            camera_definition_path=camera_definition_path,
            opct=camera.photoelectron_spectrum.opct,
            mv_per_pe=camera.photoelectron_pulse.mv_per_pe,
            pulse_width=camera.photoelectron_pulse.sigma * 2.355,
            nsb_rate=nsb_rate,
            trigger_rate=trigger_rate,
            trigger_threshold=threshold,
            pedestal=pedestal,
            pulse_area=camera.photoelectron_pulse.area,
            spectrum_average=camera.photoelectron_spectrum.average,
        )
        for event in tqdm(generator, total=len(generator.reader)):
            writer.append(event)


if __name__ == '__main__':
    main()
