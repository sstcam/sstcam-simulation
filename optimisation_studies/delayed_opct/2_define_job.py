from sstcam_simulation import Camera, PhotoelectronSource, EventAcquisition
from sstcam_simulation.performance import ChargeExtractor, LabIlluminationGenerator, \
    EventsWriter, obtain_trigger_threshold, obtain_pedestal
from sstcam_simulation.data import get_data
from spefit import ChargeContainer, minimize_with_iminuit, SiPMGentile, BinnedNLL
import numpy as np
from matplotlib import pyplot as plt
import argparse
from os.path import exists
from tqdm import tqdm


def measure_opct(camera, plot_path=None):
    print("Measuring OPCT")
    original_n_pixels = camera.mapping.n_pixels
    camera.mapping.reinitialise(1)  # Only use 1 pixel for this process
    camera.coupling.update_nsb_rate(0)  # No NSB
    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)

    illuminations = [1.0, 1.5, 2.0]
    n_illuminations = len(illuminations)
    n_events = 20000

    # Simulate waveforms and extract charge
    charge_arrays = []
    time = 20
    extractor = ChargeExtractor.from_camera(camera)  # Default extractor
    for illumination in illuminations:
        charge_array = np.zeros(n_events)
        for iev in range(n_events):
            pe = source.get_uniform_illumination(time, illumination)
            readout = acquisition.get_continuous_readout(pe)
            waveform = acquisition.get_sampled_waveform(readout)
            charge_array[iev] = extractor.extract(waveform, time)[0]
        charge_arrays.append(charge_array / camera.photoelectron_pulse.area)

    charge_containers = [
        ChargeContainer(c, n_bins=100, range_=(-1, 8))
        for c in charge_arrays
    ]

    # Fit spectra
    pdf = SiPMGentile(n_illuminations=n_illuminations)
    cost = BinnedNLL(pdf, charge_containers)
    values, errors = minimize_with_iminuit(cost)

    if plot_path is not None:
        values_array = np.array(list(values.values()))
        fig = plt.figure(figsize=(20, 10))
        x = np.linspace(-1, 8, 100000)
        for i in range(n_illuminations):
            ax = fig.add_subplot(n_illuminations, 1, i+1)
            charge = charge_containers[i]
            ax.hist(
                charge.between,
                weights=charge.hist,
                bins=charge.edges,
                histtype='step',
                density=True
            )
            fy = pdf(x, values_array, i)
            ax.plot(x, fy)
        print(f"Saving plot: {plot_path}")
        fig.savefig(plot_path)

    camera.mapping.reinitialise(original_n_pixels)
    return values['opct'], errors['opct']


def measure_50pe_pulse(camera):
    print("Measuring 50pe pulse")
    original_n_pixels = camera.mapping.n_pixels
    camera.mapping.reinitialise(1)  # Only use 1 pixel for this process
    camera.coupling.update_nsb_rate(0)  # No NSB
    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)

    time = 20
    illumination = 50
    n_events = 100
    readout = np.zeros((n_events, camera.continuous_readout_time_axis.size))
    for iev in range(n_events):
        pe = source.get_uniform_illumination(time, illumination)
        readout[iev] = acquisition.get_continuous_readout(pe)[0]
    avg = readout.mean(0)
    camera.mapping.reinitialise(original_n_pixels)
    return camera.continuous_readout_time_axis, avg


def main():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Run a camera definition file through the sstcam-simulation "
                    "chain to obtain events with a uniform lab illumination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', dest='input_path', help='Path to the camera definition file'
    )
    # parser.add_argument(
    #     '-n', dest='n_events', type=int,
    #     help='Number of events to simulate per illumination'
    # )
    parser.add_argument(
        '--nsb', default=40, type=float, dest='nsb_rate',
        help='NSB Rate to simulate (MHz)'
    )
    parser.add_argument(
        '--trigger', default=600, type=float, dest='trigger_rate',
        help='Desired camera trigger rate'
    )
    parser.add_argument(
        '--50peref', default=False, action='store_true', dest='use_50pe_ref',
        help='Use a 50pe reference pulse for the charge extraction'
    )
    args = parser.parse_args()

    camera_definition_path = args.input_path
    # n_events = args.n_events
    nsb_rate = args.nsb_rate
    trigger_rate = args.trigger_rate
    use_50pe_ref = args.use_50pe_ref

    proton_path = get_data("cherenkov/proton.h5")
    if not exists(proton_path):
        msg = '''
        Cherenkov files have not been downloaded to sstcam_simulation/data/cherenkov. 
        The files can be downloaded from Nextcloud 
        https://pcloud.mpi-hd.mpg.de/index.php/f/142621
        '''
        raise ValueError(msg)

    spe_path = camera_definition_path.replace(".pkl", "_spe.pdf")
    bias_scan_path = camera_definition_path.replace(".pkl", "_biasscan.pdf")
    events_path = camera_definition_path.replace(".pkl", "_events.h5")
    if use_50pe_ref:
        events_path = camera_definition_path.replace(".pkl", "_events_50peref.h5")

    print(f"Loading camera: {camera_definition_path}")
    camera = Camera.load(camera_definition_path)

    # Simulate just 1 superpixel
    camera.mapping.reinitialise(4)

    if use_50pe_ref:
        ref_x, ref_y = measure_50pe_pulse(camera)
        extractor = ChargeExtractor(ref_x, ref_y, camera.mapping)
    else:
        extractor = ChargeExtractor.from_camera(camera)

    # Determine calibration
    measured_opct, measured_opct_err = measure_opct(camera, spe_path)
    pedestal = obtain_pedestal(camera, extractor, nsb_rate)
    threshold = obtain_trigger_threshold(camera, nsb_rate, trigger_rate, bias_scan_path)

    camera.update_trigger_threshold(threshold)
    camera.coupling.update_nsb_rate(nsb_rate)

    generator = LabIlluminationGenerator(camera, extractor, pedestal, nsb_rate)
    with EventsWriter(events_path, generator.event_table_layout) as writer:
        writer.add_metadata(
            camera_definition_path=camera_definition_path,
            input_opct=camera.photoelectron_spectrum.opct,
            time_constant=camera.photoelectron_spectrum.time_constant,
            nsb_rate=nsb_rate,
            trigger_rate=trigger_rate,
            trigger_threshold=threshold,
            pedestal=pedestal,
            measured_opct=measured_opct,
            measured_opct_err=measured_opct_err,
        )
        illuminations = np.geomspace(0.1, 1000, 100)
        for illumination in tqdm(illuminations):
            generator.set_illumination(illumination)
            n_events = 2000 if illumination <= 30 else 100
            for _ in range(n_events):
                writer.append(generator.generate_event())



# def plot():
#     with pd.HDFStore("data/measured_opct.h5") as store:
#         df = store['data']
#
#     fig, ax = plt.subplots()
#
#     for time_constant, group in df.groupby("time_constant"):
#         group = group.sort_values('input_opct')
#         x = group['input_opct'].values
#         y = group['measured_opct'].values
#         yerr = group['measured_opct_err'].values
#
#         label = f"Average Delay = {time_constant:.0f} ns"
#         (_, caps, _) = ax.errorbar(
#             x, y, yerr=yerr, mew=1, capsize=3, elinewidth=0.7,
#             markersize=3, label=label
#         )
#         for cap in caps:
#             cap.set_markeredgewidth(0.7)
#
#     ax.legend(loc='best')
#     ax.set_xlabel("Input OCT")
#     ax.set_ylabel("Measured OCT")
#     fig.savefig("figures/measured_opct.pdf")


if __name__ == '__main__':
    main()
