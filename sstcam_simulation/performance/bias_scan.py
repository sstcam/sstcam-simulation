from sstcam_simulation import PhotoelectronSource, EventAcquisition, Camera
import numpy as np
from numpy.polynomial.polynomial import polyval, polyfit
from tqdm import trange
from matplotlib import pyplot as plt


def _calculate_expected_superpixel_rate(camera_rate, coincidence_window):
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


def _perform_sp_bias_scan(camera, nsb):
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
    original_n_pixels = camera.mapping.n_pixels
    camera.mapping.reinitialise(4)  # Only use 1 superpixel for this process
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
            thresholds /= camera.photoelectron_pulse.height  # Convert to p.e.

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

    camera.mapping.reinitialise(original_n_pixels)
    return thresholds, rate_avg, rate_err


def _get_threshold_for_rate(thresholds, trigger_rates, trigger_rates_err, requested_rate):
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


def obtain_trigger_threshold(camera, nsb_rate=40, trigger_rate=600, plot_path=None):
    print("Obtaining trigger threshold")

    # Update the AC coupling for this nsb rate
    camera.coupling.update_nsb_rate(nsb_rate)

    # Define acceptable camera trigger rate
    coincidence_window = camera.digital_trigger_length
    superpixel_rate = _calculate_expected_superpixel_rate(
        trigger_rate, coincidence_window
    )

    # Set camera threshold
    thresholds, rate_avg, rate_err = _perform_sp_bias_scan(camera, nsb_rate)
    required_threshold, gradient, y_intercept = _get_threshold_for_rate(
        thresholds, rate_avg, rate_err, superpixel_rate
    )

    # Plot bias scan
    if plot_path is not None:
        fig, ax = plt.subplots()
        label = f"Scan ({nsb_rate:.2f} MHz NSB)"
        ax.errorbar(thresholds, rate_avg, yerr=rate_err, label=label, fmt='.')
        label = f"Fit (Required threshold = {required_threshold:.2f} p.e.)"
        x = thresholds
        y = 10**polyval(thresholds, [y_intercept, gradient])
        ax.plot(x, y, label=label)
        ax.axhline(superpixel_rate, color='black', ls='-')
        ax.axvline(required_threshold, color='black', ls=':')
        ax.set_yscale('log')
        ax.set_xlabel("Threshold (p.e.)")
        ax.set_ylabel("Trigger Rate (Hz)")
        ax.legend()
        print(f"Saving plot: {plot_path}")
        fig.savefig(plot_path)

    return required_threshold
