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
        '--nsb', default=40, type=float, dest='nsb_rate',
        help='NSB Rate to simulate (MHz)'
    )
    parser.add_argument(
        '--trigger', default=600, type=float, dest='trigger_rate',
        help='Desired camera trigger rate'
    )
    args = parser.parse_args()

    pe_path = args.pe_path
    camera_definition_path = args.camera_path
    nsb_rate = args.nsb_rate
    trigger_rate = args.trigger_rate

    bias_scan_path = camera_definition_path.replace(".pkl", "_biasscan.pdf")
    events_path = camera_definition_path.replace(".pkl", "_events.h5")

    print(f"Loading camera: {camera_definition_path}")
    camera = Camera.load(camera_definition_path)

    # Simulate just 1 superpixel
    # camera.mapping.reinitialise(4)

    # Determine calibration
    extractor = ChargeExtractor.from_camera(camera)
    pedestal = obtain_pedestal(camera, extractor, nsb_rate)
    threshold = obtain_trigger_threshold(camera, nsb_rate, trigger_rate, bias_scan_path)

    camera.update_trigger_threshold(threshold)
    camera.coupling.update_nsb_rate(nsb_rate)

    generator = CherenkovShowerGenerator(
        path=pe_path,
        camera=camera,
        extractor=extractor,
        pedestal=pedestal,
        nsb_rate=nsb_rate
    )
    with EventsWriter(events_path, generator.event_table_layout) as writer:
        writer.add_metadata(
            camera_definition_path=camera_definition_path,
            self_opct=camera.photoelectron_spectrum.opct,
            reflected_opct=camera.photoelectron_spectrum.reflected_opct,
            reflected_scale=camera.photoelectron_spectrum.reflected_scale,
            nsb_rate=nsb_rate,
            trigger_rate=trigger_rate,
            trigger_threshold=threshold,
            pedestal=pedestal,
        )
        for event in tqdm(generator, total=len(generator.reader)):
            writer.append(event)


if __name__ == '__main__':
    main()
