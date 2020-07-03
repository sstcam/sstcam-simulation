from sstcam_simulation import PhotoelectronSource, EventAcquisition, Camera
import argparse
import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
import tables
from scipy import interpolate
from scipy.ndimage import correlate1d


def calculate_expected_superpixel_rate(camera_rate, coincidence_window):
    """
    Calculate the expected superpixel trigger rate for a given camera trigger
    rate using statistical evaluation of coincident triggers:
    http://courses.washington.edu/phys433/muon_counting/statistics_tutorial.pdf

    Parameters
    ----------
    camera_rate : float
        The camera trigger rate. Unit: Hz
    coincidence_window : float
        Maximum delay allowed between neighbouring superpixel triggers
        (threshold crossings) for a camera trigger to be generated
        Unit: nanoseconds

    Returns
    -------
    float
    """
    n_superpixel_neighbours = 1910  # (from camera geometry)
    n_combinations = 2 * n_superpixel_neighbours
    return np.sqrt(camera_rate / (n_combinations * coincidence_window * 1e-9))


def perform_nsb_bias_scan(camera, nsb):
    """
    While observing NSB only, scan through trigger threshold and measure
    superpixel trigger rate

    Parameters
    ----------
    camera : Camera
        Description of the camera
    nsb : float
        NSB rate (Unit: MHz)

    Returns
    -------
    thresholds : ndarray
        Thresholds sampled
    mean_rate : ndarray
        Average trigger rate at each threshold
    std_rate : ndarray
        Standard deviation of trigger rate at each threshold

    """
    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    trigger = acquisition.trigger
    n_repeats = 10000  # Repeats for statistics
    n_thresholds = 50
    thresholds = None  # Delayed initialisation based on readout limits
    n_triggers = np.zeros((n_repeats, n_thresholds))
    for iev in trange(n_repeats, desc="Measuring bias curve"):
        photoelectrons = source.get_nsb(nsb)
        readout = acquisition.get_continuous_readout(photoelectrons)

        # Define thresholds based on waveform readout
        if thresholds is None:
            min_ = readout.min()
            min_ = min_ if min_ >= 0 else 0
            max_ = readout.max() * 5
            max_ = max_ if max_ > 0 else 1
            thresholds = np.linspace(min_, max_, n_thresholds)
            thresholds /= camera.reference_pulse.peak_height  # Convert to p.e.

        # Scan through thresholds to count triggers
        for i, thresh in enumerate(thresholds):
            camera.update_trigger_threshold(thresh)
            digital_trigger = trigger.get_superpixel_digital_trigger_line(readout)
            n_triggers[iev, i] = trigger.get_n_superpixel_triggers(digital_trigger)

    n_triggers[:, n_triggers.sum(0) <= 1] = np.nan  # Remove entries with low statistics
    n_triggers_avg = n_triggers.mean(0)
    n_triggers_err = np.sqrt(n_triggers_avg / n_repeats)
    rate_avg = n_triggers_avg / (camera.continuous_readout_duration * 1e-9)
    rate_err = n_triggers_err / (camera.continuous_readout_duration * 1e-9)
    return thresholds, rate_avg, rate_err


def get_threshold_for_rate(thresholds, trigger_rates, trigger_rates_err, requested_rate):
    """
    Obtain the trigger threshold that corresponds to a specific superpixel
    trigger rate (from the NSB bias scan)

    Parameters
    ----------
    thresholds
    trigger_rates
    requested_rate

    Returns
    -------

    """
    # Select the last 10 valid points (where the bias curve is likely linear)
    mask = np.isfinite(trigger_rates)
    thresholds = thresholds[mask][-10:]
    rates = trigger_rates[mask][-10:]
    err = trigger_rates_err[mask][-10:]

    y_intercept, gradient = polyfit(thresholds, np.log10(rates), 1, w=1/np.log10(err))
    required_threshold = (np.log10(requested_rate) - y_intercept) / gradient
    return required_threshold, gradient, y_intercept


class ChargeExtractor:
    def __init__(self, camera, peak_index):
        self.peak_index = peak_index

        ref_x = camera.reference_pulse.time
        ref_y = camera.reference_pulse.pulse

        # Prepare for cc
        f = interpolate.interp1d(ref_x, ref_y, kind=3)
        cc_ref_x = np.arange(0, ref_x[-1], 1)
        cc_ref_y = f(cc_ref_x)
        y_1pe = cc_ref_y / np.trapz(cc_ref_y)
        self.origin = cc_ref_y.argmax() - cc_ref_y.size // 2
        scale = correlate1d(y_1pe, cc_ref_y, mode='constant', origin=self.origin).max()
        self.cc_ref_y = cc_ref_y / scale

    def extract(self, waveforms):
        cc = correlate1d(waveforms, self.cc_ref_y, mode='constant', origin=self.origin)
        charge = cc[:, self.peak_index]
        return charge


FILTERS = tables.Filters(
    complevel=5,  # compression medium, tradeoff between speed and compression
    complib="blosc:zstd",  # use modern zstd algorithm
    fletcher32=True,  # add checksums to data chunks
)


def main():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Run a camera definition file through the sstcam-simulation "
                    "chain to obtain events with a uniform lab illumination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', dest='input_path', help='path to the camera definition file'
    )
    parser.add_argument(
        '--nsb', default=40, type=float, dest='nsb_rate',
        help='NSB Rate to simulate (MHz)'
    )
    args = parser.parse_args()

    camera_definition_path = args.input_path
    nsb_rate = args.nsb_rate

    # Define output paths
    biasscan_path = camera_definition_path.replace(".pkl", "_biasscan.pdf")
    events_path = camera_definition_path.replace(".pkl", "_lab.h5")

    # Load the camera from the path
    camera = Camera.load(camera_definition_path)
    n_pixels = 4  # Only use 1 superpixel for this investigation
    camera.mapping.reinitialise(n_pixels)

    # Define acceptable camera trigger rate
    camera_rate = 600  # Hz (from SST requirements)
    coincidence_window = camera.digital_trigger_length
    superpixel_rate = calculate_expected_superpixel_rate(camera_rate, coincidence_window)

    # Set camera threshold
    thresholds, rate_avg, rate_err = perform_nsb_bias_scan(camera, nsb_rate)
    required_threshold, gradient, y_intercept = get_threshold_for_rate(
        thresholds, rate_avg, rate_err, superpixel_rate
    )
    camera.update_trigger_threshold(required_threshold)

    # Plot bias scan
    fig, ax = plt.subplots()
    label = f"Scan ({nsb_rate:.2f} MHz NSB)"
    ax.errorbar(thresholds, rate_avg, yerr=rate_err, label=label, fmt='.')
    label = f"Fit (Required threshold = {required_threshold:.2f} p.e.)"
    ax.plot(thresholds, 10**polyval(thresholds, [y_intercept, gradient]), label=label)
    ax.axhline(superpixel_rate, color='black', ls='-')
    ax.axvline(required_threshold, color='black', ls=':')
    ax.set_yscale('log')
    ax.set_xlabel("Threshold (p.e.)")
    ax.set_ylabel("Trigger Rate (Hz)")
    ax.legend()
    print(f"Saving plot: {biasscan_path}")
    fig.savefig(biasscan_path)

    # Illumination scan
    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    trigger = acquisition.trigger
    signal_time = 60  # ns
    signal_index = camera.get_waveform_sample_from_time(signal_time)
    extractor = ChargeExtractor(camera, signal_index)
    illuminations = np.geomspace(0.1, 1000, 100)

    # Calculate pedestal
    n_empty = 10000
    pedestal_array = np.zeros((n_empty, n_pixels))
    for i in trange(n_empty, desc="Measuring pedestal"):
        nsb_pe = source.get_nsb(nsb_rate)
        readout = acquisition.get_continuous_readout(nsb_pe)
        waveform = acquisition.get_sampled_waveform(readout)
        charge = extractor.extract(waveform)
        pedestal_array[i] = charge
    pedestal = np.median(pedestal_array)
    print(f"Pedestal = {pedestal:.3f}")

    class EventTable(tables.IsDescription):
        illumination = tables.Float64Col()
        n_triggers = tables.Int64Col()
        true_charge = tables.Int64Col(shape=camera.mapping.n_pixels)
        measured_charge = tables.Float64Col(shape=camera.mapping.n_pixels)

    print(f"Creating file: {events_path}")
    with tables.File(events_path, mode='w', filters=FILTERS) as h5_file:
        table = h5_file.create_table(h5_file.root, "event", EventTable, "Events")

        for illumination in tqdm(illuminations, desc="Illumination scan"):
            n_simulated = 2000 if illumination <= 30 else 100
            for _ in range(n_simulated):
                nsb_pe = source.get_nsb(nsb_rate)
                laser_pe = source.get_uniform_illumination(
                    signal_time, illumination, laser_pulse_width=3
                )
                readout = acquisition.get_continuous_readout(nsb_pe + laser_pe)

                digital_trigger = trigger.get_superpixel_digital_trigger_line(readout)
                n_triggers = trigger.get_n_superpixel_triggers(digital_trigger)

                waveform = acquisition.get_sampled_waveform(readout)
                charge = extractor.extract(waveform) - pedestal

                event = table.row
                event['illumination'] = illumination
                event['n_triggers'] = n_triggers
                event['true_charge'] = laser_pe.get_photoelectrons_per_pixel(n_pixels)
                event['measured_charge'] = charge
                event.append()


if __name__ == '__main__':
    main()
