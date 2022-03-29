from sstcam_simulation import PhotoelectronSource, EventAcquisition, Camera, PhotoelectronReader
from ctapipe.io import SimTelEventSource
import astropy.units as u
import numpy as np
from numpy.polynomial.polynomial import polyval, polyfit
from tqdm import trange, tqdm
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
    for iev in trange(n_repeats, desc="Scanning NSB triggers"):
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


def _perform_nsb_bias_scan(camera, nsb_rate):
    """
    While observing NSB only, scan through trigger threshold and measure
    camera trigger rate

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
    # Determine conversion from superpixel to camera trigger rate
    coincidence_window = camera.digital_trigger_length
    trigger_rate = np.logspace(1,8,10)
    superpixel_rate = _calculate_expected_superpixel_rate(
        trigger_rate, coincidence_window
    )
    n, m = polyfit(np.log10(superpixel_rate), np.log10(trigger_rate), 1)

    # Perform bias scan for NSB
    thresholds, sp_rate_avg, sp_rate_err = _perform_sp_bias_scan(camera, nsb_rate)
    cam_rate_avg = 10**polyval(np.log10(sp_rate_avg), [n,m])
    cam_rate_err = sp_rate_err / sp_rate_avg * cam_rate_avg
    return thresholds, cam_rate_avg, cam_rate_err


def _fit_nsb_bias_scan(thresholds, rate_avg, rate_err):
    """
    Fit NSB bias curve

    Parameters
    ----------
    thresholds : ndarray
        Thresholds sampled
    rate : ndarray
        Trigger rate at each threshold
    rate_err : ndarray
        Standard deviation of trigger rate at each threshold

    Returns
    -------
    y_intercept: float
        y intercept of fit
    gradient: float
        gradient of fit
    """ 
    # Select the last 10 valid points (where the bias curve is likely linear)
    mask = np.isfinite(rate_avg)
    x = thresholds[mask][-12:-2]
    y = rate_avg[mask][-12:-2]
    yerr = rate_err[mask][-10:]
    y_intercept, gradient = polyfit(x, np.log10(y), 1, w=1/np.log10(yerr))
    return y_intercept, gradient


def _perform_cr_bias_scan(camera, pe_path):
    """
    While observing NSB only, scan through trigger threshold and measure
    camera trigger rate

    Parameters
    ----------
    camera : Camera
        Description of the camera
    pe_path : string
        Path to the photoelectrons file

    Returns
    -------
    thresholds : ndarray
        Thresholds sampled
    rate : ndarray
        Trigger rate at each threshold
    std_rate : ndarray
        Standard deviation of trigger rate at each threshold
    """    
    reader = PhotoelectronReader(pe_path)
    
    # Get energies from events of one telescope
    energies = []
    for pe in reader:
        if pe.metadata['telescope_id'] == 1:
            energies += [pe.metadata['energy']]
            
    # Get simtel information
    try:
        events = SimTelEventSource(
                input_url=pe_path.replace('_pe.h5', '.simtel'),
                max_events=1, back_seekable=True
            )
    except:
        print('Please provide the corresponding .simtel file in the same folder as the .h5 file.')
        
    Emin = events.mc_header['energy_range_min']
    Emax = events.mc_header['energy_range_max']
    index_sim = events.mc_header['spectral_index']
    num_showers = events.mc_header['num_showers']
    shower_reuse = events.mc_header['shower_reuse']
    total_number_showers = num_showers * shower_reuse
    view = events.mc_header['max_viewcone_radius']
    omega = (1 - np.cos(view)) * 2 * np.pi * u.sr
    scatter = events.mc_header['max_scatter_range']
    area = np.pi * scatter ** 2
    
    # Determine weights
    Eref = 1 * u.TeV
    T = 1 * u.s
    norm_sim = Eref**index_sim * (index_sim + 1) * total_number_showers / (T * omega * area * (Emax**(index_sim+1) - Emin**(index_sim+1)))
    
    # BESS Proton spectrum  ApJ 545, 1135 (2000) [arXiv:astro-ph/0002481],
    # same as used by K. Bernloehr.
    index_cr = -2.7
    norm_cr = 9.6e-9 / (u.GeV * u.cm**2 * u.s * u.sr)
    flux_cr = norm_cr * (energies*u.TeV / Eref) ** index_cr
    flux_sim = norm_sim * (energies*u.TeV / Eref) ** index_sim
    
    weights = (flux_cr / flux_sim).decompose()
            
    # perform bias scan   
    n_thresholds = 5
    n_events = len(energies)
    thresholds = np.linspace(6,15,n_thresholds)
    triggers = np.zeros((n_events, n_thresholds))
    
    for i, thresh in enumerate(tqdm(thresholds, leave=True, position=0, desc='Scanning proton triggers')):
        camera.update_trigger_threshold(thresh)
        source = PhotoelectronSource(camera=camera)
        acquisition = EventAcquisition(camera=camera)

        j = 0
        for pe in reader:
            if pe.metadata['telescope_id'] == 1:
                signal_pe = source.resample_photoelectron_charge(pe)
                readout = acquisition.get_continuous_readout(signal_pe)
                n_triggers = acquisition.get_trigger(readout).size

                if n_triggers > 0:
                    triggers[j, i] = 1

                j = j+1
    
    # Calculate proton rate
    rate = np.sum(weights[:,None] * triggers, axis=0)
    rate_err = rate / np.sqrt(np.sum(triggers, axis=0))     
    return thresholds, rate, rate_err


def _fit_cr_bias_scan(thresholds, rate, rate_err):
    """
    Fit proton bias curve

    Parameters
    ----------
    thresholds : ndarray
        Thresholds sampled
    rate : ndarray
        Trigger rate at each threshold
    rate_err : ndarray
        Standard deviation of trigger rate at each threshold

    Returns
    -------
    y_intercept: float
        y intercept of fit
    gradient: float
        gradient of fit
    """ 
    mask = np.isfinite(rate)
    x = thresholds[mask]
    y = rate[mask]
    yerr = rate_err[mask]
    y_intercept, gradient = polyfit(x, np.log10(y), 1, w=1/np.log10(yerr))
    return y_intercept, gradient    


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


def obtain_safe_trigger_threshold(camera, pe_path, nsb_rate=40, plot_path=None):
    """
    Obtain "safe" trigger threshold, defined as the intercept between the NSB and 1.5 x proton curve.

    Parameters
    ----------
    camera : Camera
        Description of the camera
    pe_path : string
        Path to the photoelectrons file
    nsb : float
        NSB rate (Unit: MHz)
    plot_path : string
        Path for saving the plot

    Returns
    -------
    threshold: float
        Safe threshold
    """ 
    print("Obtaining safe trigger threshold")

    # Update the AC coupling for this nsb rate
    camera.coupling.update_nsb_rate(nsb_rate)
    
    # Determine NSB curve
    thresholds_nsb, cam_rate_nsb_avg, cam_rate_nsb_err = _perform_nsb_bias_scan(camera, nsb_rate)
    y_intercept_nsb, gradient_nsb = _fit_nsb_bias_scan(thresholds_nsb, cam_rate_nsb_avg, cam_rate_nsb_err)
    
    
    ## Determine 1.5 x proton curve
    thresholds_cr, cam_rate_cr, cam_rate_cr_err = _perform_cr_bias_scan(camera, pe_path)
    cam_rate_cr *= 1.5
    cam_rate_cr_err *= 1.5
    y_intercept_cr, gradient_cr = _fit_cr_bias_scan(thresholds_cr, cam_rate_cr, cam_rate_cr_err)
    
    
    required_threshold = (y_intercept_cr - y_intercept_nsb) \
                            / (gradient_nsb - gradient_cr)
    resulting_rate = 10**(gradient_cr * required_threshold + y_intercept_cr)
    
    # Plot bias scan
    if plot_path is not None:    
        plt.errorbar(thresholds_nsb, cam_rate_nsb_avg, yerr=cam_rate_nsb_err, 
                     label=f'NSB: {nsb_rate} MHz', fmt='.', color='darkblue')
        
        plt.plot(thresholds_nsb, 10**polyval(thresholds_nsb, [y_intercept_nsb, gradient_nsb]), 
                                         color='darkblue', alpha=0.5)
        
        plt.errorbar(thresholds_cr, cam_rate_cr, yerr=cam_rate_cr_err,
                    label=f'1.5 x Proton', fmt='.', color='red')
        
        plt.plot(thresholds_cr, 10**polyval(thresholds_cr, [y_intercept_cr, gradient_cr]), 
                                         color='red', alpha=0.5)

        plt.axhline(600, color='k', ls=':', label='600 Hz', lw=0.5)
        
        plt.axvline(required_threshold, color='k', ls='--', alpha=0.8, 
                    label=f'{required_threshold:.2f} p.e. @ {resulting_rate:.0f} Hz'
                   )        

        plt.xlabel('Threshold / p.e.')
        plt.xlim(0,20)

        plt.ylabel('Camera Trigger Rate / Hz')
        plt.ylim(1,1e8)
        plt.yscale('log')

        plt.legend()

        plt.savefig(plot_path)

    return required_threshold
