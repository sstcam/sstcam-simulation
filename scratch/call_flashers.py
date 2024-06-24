import sys

sys.path.append(r'/home/sheridan/sstcam-simulation') # change this to your appropriate local root to run this code
#import /home/sheridan/sstcam-simulation/sstcam_simulation

from sstcam_simulation.event.source import PhotoelectronSource
from sstcam_simulation.camera import Camera
from sstcam_simulation.camera import SSTCameraMapping
from sstcam_simulation.event.acquisition import EventAcquisition
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.plotting.image import CameraImage
from matplotlib import pyplot as plt
import numpy as np
import performance

# Define the camera
camera = Camera(
    trigger_threshold=20,
    readout_noise=GaussianNoise(stddev=0.1),
    lookback_time=15,
    n_waveform_samples=600
)


def multiple_flasher_call(start_time, illumination, flasher_pulse_width, illumination_err, pulse_width_err, end_time, interval_nanosec):

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    flasher = source.get_flasher_illumination(time=start_time, illumination=illumination,
                                              flasher_pulse_width=flasher_pulse_width,
                                              illumination_err=illumination_err, pulse_width_err=pulse_width_err)
    combined_pe = flasher

    for time in np.arange(start_time + interval_nanosec, end_time , interval_nanosec):
        source = PhotoelectronSource(camera=camera)
        acquisition = EventAcquisition(camera=camera)
        flasher = source.get_flasher_illumination(time=time, illumination=illumination,
                                                  flasher_pulse_width=flasher_pulse_width,
                                                  illumination_err=illumination_err, pulse_width_err=pulse_width_err)
        combined_pe = combined_pe + flasher

    return combined_pe


def single_flasher_call_no_err():

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)

    flasher = source.get_flasher_illumination(time= 0, illumination=500, flasher_pulse_width = 4.5  )
    pe = flasher

    readout = acquisition.get_continuous_readout(pe)
    trigger_times = acquisition.get_trigger(readout)
    waveform = acquisition.get_sampled_waveform(readout, trigger_time=trigger_times[0])

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    image = CameraImage.from_coordinates(camera.mapping.pixel, ax=ax1)
    image.add_colorbar("Amplitude (p.e.)")
    image.image = waveform.max(1) / camera.photoelectron_pulse.height
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude (p.e.)')
    ax2.plot(waveform.T)
    plt.show()
    return


def single_flasher_call_with_err(time, illumination, flasher_pulse_width, illumination_err, pulse_width_err):

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    flasher = source.get_flasher_illumination(time=time, illumination=illumination, flasher_pulse_width=flasher_pulse_width, illumination_err=illumination_err, pulse_width_err=pulse_width_err)
    pe = flasher
    readout = acquisition.get_continuous_readout(pe)
    trigger_times = acquisition.get_trigger(readout)
    waveform = acquisition.get_sampled_waveform(readout, trigger_time=trigger_times[0])
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    image = CameraImage.from_coordinates(camera.mapping.pixel, ax=ax1)
    image.add_colorbar("Amplitude (p.e.)")
    image.image = waveform.max(1) / camera.photoelectron_pulse.height
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude (p.e.)')
    ax2.plot(waveform.T)
    plt.show()

    return

# Uncomment these to see a single flasher pulse waveform
#single_flasher_call_no_err()
#single_flasher_call_with_err(time=2, illumination=500, flasher_pulse_width=4.5, illumination_err=0, pulse_width_err=0)
#single_flasher_call_with_err(time=2, illumination=500, flasher_pulse_width=4.5, illumination_err=0.2, pulse_width_err=0.035)
#single_flasher_call_with_err(time=2, illumination=500, flasher_pulse_width=4.5, illumination_err=0.5, pulse_width_err=0.035)
#single_flasher_call_with_err(time=2, illumination=500, flasher_pulse_width=4.5, illumination_err=0.5, pulse_width_err=0.2)


def plot_multiple_flasher(illumination_err, pulse_width_err):

    acquisition = EventAcquisition(camera=camera)
    pe = multiple_flasher_call(start_time=0, illumination=1000, flasher_pulse_width=4.5, illumination_err=illumination_err, pulse_width_err=pulse_width_err, end_time=1000, interval_nanosec=100)
    readout = acquisition.get_continuous_readout(pe)
    trigger_times = acquisition.get_trigger(readout)
    waveform = acquisition.get_sampled_waveform(readout, trigger_time=trigger_times[0])
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    image = CameraImage.from_coordinates(camera.mapping.pixel, ax=ax1)
    image.add_colorbar("Amplitude (p.e.)")
    image.image = waveform.max(1) / camera.photoelectron_pulse.height
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude (p.e.)')
    ax2.plot(waveform.T)
    if illumination_err==0 and pulse_width_err==0:
        plt.title("Multiple Flasher - no err - pulse width 4.5 ns, illumination 1000 p.e. ")
    else:
        plt.title("Multiple Flasher - pulse width 4.5 ns err " + "%.1f" % (pulse_width_err*100) + " %, illumination 1000 p.e. err: " + "%.0f" % (illumination_err*100) +" %")
    plt.show()
    return

#Uncomment this to see multiple flasher waveforms which vary within error bounds or with no error applied
#print ("Running plot_multiple_flasher(illumination_err=0.2, pulse_width_err=0.035)")
#plot_multiple_flasher(illumination_err=0, pulse_width_err=0)
#plot_multiple_flasher(illumination_err=0.2, pulse_width_err=0.035)
#plot_multiple_flasher(illumination_err=0, pulse_width_err=0)


# test of get uniform illumination - was altered for flashers to accept a pulse width err
def plot_uniform_illumination(time, illumination, laser_pulse_width, pulse_width_err):

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    flasher = source.get_uniform_illumination(time, illumination, laser_pulse_width, pulse_width_err)
    pe = flasher
    readout = acquisition.get_continuous_readout(pe)
    trigger_times = acquisition.get_trigger(readout)
    waveform = acquisition.get_sampled_waveform(readout, trigger_time=trigger_times[0])
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    image = CameraImage.from_coordinates(camera.mapping.pixel, ax=ax1)
    image.add_colorbar("Amplitude (p.e.)")
    image.image = waveform.max(1) / camera.photoelectron_pulse.height
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude (p.e.)')
    ax2.plot(waveform.T)
    plt.show()
    return

# Uncomment this to see a uniform illumination pulse
#plot_uniform_illumination(time=200, illumination=500, laser_pulse_width=0, pulse_width_err=0)
#plot_uniform_illumination(time=200, illumination=500, laser_pulse_width=40, pulse_width_err=0)
#plot_uniform_illumination(time=200, illumination=500, laser_pulse_width=3, pulse_width_err=5)

sys.path.append(r'/home/sheridan/sstcam-simulation/src/checlabpy/CHECLabPy')

#
# FLASHER CHARGE RESOLUTION TEST CODE
#

from CHECLabPy.utils.resolutions import ChargeResolution
from tqdm import tqdm, trange

# Define the camera
camera = Camera(
    mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
    continuous_readout_duration=128,  # Only need a single-waveform's worth of readout
    n_waveform_samples=128,
)
n_pixels = camera.mapping.n_pixels
pixel_array = np.arange(n_pixels)

from sstcam_simulation.camera import SSTCameraMapping
from tqdm import tqdm, trange

def bin_dataframe(df, n_bins=40):
    true = df['true'].values
    min_ = true.min()
    max_ = (true.max() // 500 + 1) * 500
    bins = np.geomspace(0.1, max_, n_bins)
    bins = np.append(bins, 10**(np.log10(bins[-1]) + np.diff(np.log10(bins))[0]))
    df['bin'] = np.digitize(true, bins, right=True) - 1

    log = np.log10(bins)
    between = 10**((log[1:] + log[:-1]) / 2)
    edges = np.repeat(bins, 2)[1:-1].reshape((bins.size-1 , 2))
    edge_l = edges[:, 0]
    edge_r = edges[:, 1]
    df['between'] = between[df['bin']]
    df['edge_l'] = edge_l[df['bin']]
    df['edge_r'] = edge_r[df['bin']]

    return df



#FlasherChargeResolutionFluctuatingNSB()

def FlasherChargeResolutionVaryTime():

    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)
    #for n_events in range(0,100,50):
    n_events=1000
    for flasher_time_var in np.arange(1,10,2):
        for illumination in tqdm(np.logspace(1, 3, 10)): # up to 1000 illumination in log space
            for iev in range(n_events):
                flasher = source.get_flasher_illumination(time=flasher_time_var, illumination=illumination,
                                                          flasher_pulse_width=4.5,
                                                          illumination_err=0.2,
                                                          pulse_width_err=0.035)
                #nsb = source.get_nsb(rate=100)
                #pe = flasher + nsb
                pe = flasher
                # Add in NSB

                #pe = source.get_uniform_illumination(time=60, illumination=illumination)
                readout = acquisition.get_continuous_readout(pe)
                waveform = acquisition.get_sampled_waveform(readout)

                # Charge Extraction
                measured_charge = waveform.sum(1)

                true_charge = pe.get_photoelectrons_per_pixel(n_pixels)
                charge_resolution.add(pixel_array, true_charge, measured_charge)

        df, _ = charge_resolution.finish()
        df = bin_dataframe(df, n_bins=40)
        df_mean = df.groupby('bin').mean()
        bin_ = df_mean.index
        x = df_mean['true'].values
        y = df_mean['charge_resolution'].values
        poisson_limit = np.sqrt(x) / x
        enf_limit = np.sqrt(camera.photoelectron_spectrum.excess_noise_factor * x) / x
        plt.plot(x, poisson_limit, label="Poisson Limit " )
        plt.plot(x, enf_limit, label="ENF Limit ", color='black')
        plt.plot(x, y, '.', label="From Waveform, Flasher Time " + str(flasher_time_var) )


    plt.legend(loc="best")
    plt.xlabel("Number of Photoelectrons")
    plt.ylabel(r"Fractional Charge Resolution $\frac{{\sigma_Q}}{{Q}}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    return

#FlasherChargeResolutionVaryTime()

def FlasherChargeResolutionNSBTrueMeasured():

    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)
    n_events=1000
    #for n_events in range(0,100,50):

    # Just flashers no NSB
    #
    TrueFlasherCharge = []
    MeasuredFlasherCharge = []
    PoissonLimit=[]

    for illumination in tqdm(np.logspace(1, 3, 10)):  # up to 1000 illumination in log space
        TrueChargeSum = 0.0
        MeasuredChargeSum = 0.0

        for iev in range(n_events):
            flasher = source.get_flasher_illumination(time=50, illumination=illumination,
                                                      flasher_pulse_width=4.5,
                                                      illumination_err=0.2,
                                                      pulse_width_err=0.035)
            pe = flasher
            readout = acquisition.get_continuous_readout(pe)
            waveform = acquisition.get_sampled_waveform(readout)
            # Charge Extraction
            measured_charge = waveform.sum(1)
            MeasuredChargeSum+=measured_charge

            true_charge = pe.get_photoelectrons_per_pixel(n_pixels)
            TrueChargeSum+=true_charge

            #PoissonLimit.append(np.sqrt(true_charge)/true_charge)
        MeasuredFlasherCharge.append(MeasuredChargeSum/n_events)
        TrueFlasherCharge.append(TrueChargeSum/n_events)
    # poisson_limit = np.sqrt(x) / x
    # enf_limit = np.sqrt(camera.photoelectron_spectrum.excess_noise_factor * x) / x
    # plt.plot(x, poisson_limit, label="Poisson Limit NSB " + str(nsb_MHz) + " MHz")
    # plt.plot(x, enf_limit, label="ENF Limit NSB "  + str(nsb_MHz) + " MHz", color='black')
    plt.plot(TrueFlasherCharge, MeasuredFlasherCharge, '-', label="Flasher 4.5 ns")
    #plt.plot(TrueFlasherCharge,PoissonLimit,'-',label="Poisson Limit")
    for nsb_MHz in [50, 100, 200, 300]:
        MyTrueCharge = []
        MyMeasuredCharge = []
        for illumination in tqdm(np.logspace(1, 3, 10)): # up to 1000 illumination in log space
            TrueChargeSum = 0.0
            MeasuredChargeSum = 0.0
            for iev in range(n_events):
                flasher = source.get_flasher_illumination(time=50, illumination=illumination,
                                                          flasher_pulse_width=4.5,
                                                          illumination_err=0.2,
                                                          pulse_width_err=0.035)
                # Add in NSB
                nsb = source.get_nsb(rate=nsb_MHz)
                pe = flasher + nsb

                #pe = source.get_uniform_illumination(time=60, illumination=illumination)

                readout = acquisition.get_continuous_readout(pe)
                waveform = acquisition.get_sampled_waveform(readout)

                # Charge Extraction
                measured_charge = waveform.sum(1)
                MeasuredChargeSum += measured_charge
                true_charge = pe.get_photoelectrons_per_pixel(n_pixels)
                TrueChargeSum += true_charge
            MyMeasuredCharge.append(MeasuredChargeSum / n_events)
            MyTrueCharge.append(TrueChargeSum / n_events)

        #poisson_limit = np.sqrt(x) / x
        #enf_limit = np.sqrt(camera.photoelectron_spectrum.excess_noise_factor * x) / x
        #plt.plot(x, poisson_limit, label="Poisson Limit NSB " + str(nsb_MHz) + " MHz")
        #plt.plot(x, enf_limit, label="ENF Limit NSB "  + str(nsb_MHz) + " MHz", color='black')
        plt.plot(MyTrueCharge, MyMeasuredCharge, '.', label="Flasher + NSB " + str(nsb_MHz) + " MHz")
    plt.plot([0,1000],[0,1000],'-',label = "True=Measured")
    plt.legend(loc="best")
    plt.xlabel("True Charge (p.e)")
    plt.ylabel("Measured Charge (p.e.)")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    return

#FlasherChargeResolutionNSBTrueMeasured()
# -------------
# 24th May 2021
# -------------
def FlasherChargeResolutionVaryTimePlotWaveForm():

    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)
    #for n_events in range(0,100,50):
    n_events=1

    #readouts=[]
    waveforms=[]
    legends=[]
    for flasher_time_var in np.arange(1,100,20):
        illumination=1000
        #for illumination in tqdm(np.logspace(1, 3, 10)): # up to 1000 illumination in log space
        for iev in range(n_events):
            flasher = source.get_flasher_illumination(time=flasher_time_var, illumination=illumination,
                                                      flasher_pulse_width=4.5,
                                                      illumination_err=0.2,
                                                      pulse_width_err=0.035)
            #nsb = source.get_nsb(rate=100)
            #pe = flasher + nsb
            pe = flasher
            # Add in NSB

            #pe = source.get_uniform_illumination(time=60, illumination=illumination)
            readout = acquisition.get_continuous_readout(pe)
            waveform = acquisition.get_sampled_waveform(readout)

            #readouts.append(readout)
            waveforms.append(waveform)

            # Charge Extraction
            measured_charge = waveform.sum(1)
            true_charge = pe.get_photoelectrons_per_pixel(n_pixels)
            legends.append("Start " + str(flasher_time_var) + " mc/tc " + str(measured_charge/true_charge) + " mc " + str(measured_charge) + ", tc " + str(true_charge))




    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude (p.e.)')
    image = CameraImage.from_coordinates(camera.mapping.pixel, ax=ax1)
    image.add_colorbar("Amplitude (p.e.)")
    image.image = waveform.max(1) / camera.photoelectron_pulse.height
    for waveform,legend in zip(waveforms,legends):
        ax2.plot(waveform.T,label=legend)

    ax2.legend()
    plt.show()


    return
# Demonstrates wrap-around behaviour of flasher waveform when started at time zero
#print ("FlasherChargeResolutionVaryTimePlotWaveForm")
#FlasherChargeResolutionVaryTimePlotWaveForm()
#
# 26th May 2021
#
def FlasherChargeResolutionNSBasErrTrueMeasured():

    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)
    n_events=1000
    #for n_events in range(0,100,50):

    # Just flashers no NSB
    #
    TrueFlasherCharge = []
    MeasuredFlasherCharge = []
    PoissonLimit=[]

    for illumination in tqdm(np.logspace(1, 4, 10)):  # up to 1000 illumination in log space
        TrueChargeSum = 0.0
        MeasuredChargeSum = 0.0

        for iev in range(n_events):
            flasher = source.get_flasher_illumination(time=50, illumination=illumination,
                                                      flasher_pulse_width=4.5,
                                                      illumination_err=0.2,
                                                      pulse_width_err=0.035)
            pe = flasher
            readout = acquisition.get_continuous_readout(pe)
            waveform = acquisition.get_sampled_waveform(readout)
            # Charge Extraction
            measured_charge = waveform.sum(1)
            MeasuredChargeSum+=measured_charge

            true_charge = pe.get_photoelectrons_per_pixel(n_pixels)
            TrueChargeSum+=true_charge


        MeasuredFlasherCharge.append(MeasuredChargeSum/n_events)
        TrueFlasherCharge.append(TrueChargeSum/n_events)

    plt.plot(TrueFlasherCharge, MeasuredFlasherCharge, '-', label="Flasher 4.5 ns")

    nsbs=[]
    illuminations=[]
    truecharges=[]
    measuredcharges=[]

    for nsb_MHz in [50, 100, 200, 300,1000]:
        MyTrueCharge = []
        MyMeasuredCharge = []
        for illumination in tqdm(np.logspace(1, 4, 10)): # up to 1000 illumination in log space
            TrueChargeSum = 0.0
            MeasuredChargeSum = 0.0
            for iev in range(n_events):
                flasher = source.get_flasher_illumination(time=50, illumination=illumination,
                                                          flasher_pulse_width=4.5,
                                                          illumination_err=0.2,
                                                          pulse_width_err=0.035)

                true_charge = flasher.get_photoelectrons_per_pixel(n_pixels)
                TrueChargeSum += true_charge

                # Add in NSB
                nsb = source.get_nsb(rate=nsb_MHz)
                pe = flasher + nsb



                readout = acquisition.get_continuous_readout(pe)
                waveform = acquisition.get_sampled_waveform(readout)

                # Charge Extraction
                measured_charge = waveform.sum(1)
                MeasuredChargeSum += measured_charge

            MyMeasuredCharge.append(MeasuredChargeSum / n_events)
            MyTrueCharge.append(TrueChargeSum / n_events)

            nsbs.append(nsb_MHz)
            illuminations.append(illumination)
            truecharges.append(TrueChargeSum / n_events)
            measuredcharges.append(MeasuredChargeSum / n_events)


        plt.plot(MyTrueCharge, MyMeasuredCharge, '-', label="Flasher + NSB " + str(nsb_MHz) + " MHz")
    plt.plot([0,10000],[0,10000],'-',label = "True=Measured")
    plt.legend(loc="best")
    plt.xlabel("True Charge Expected From Flasher (p.e)")
    plt.ylabel("Measured Charge (p.e.)")

    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    plt.clf() # start again

    for nsb_MHz in [50, 100, 200, 300, 1000]: # select and plot appropriate NSB
        plot_charge_errs=[]
        plot_illumination=[]
        for nsb, illumination, truecharge, measuredcharge in zip(nsbs, illuminations, truecharges, measuredcharges):
            if nsb==nsb_MHz:
                plot_illumination.append(illumination)
                plot_charge_errs.append(100*(measuredcharge-truecharge)/truecharge)
        plt.plot(plot_illumination, plot_charge_errs, '-', label="Flasher + NSB " + str(nsb_MHz) + " MHz")


    plt.legend(loc="best")
    plt.xlabel("Illumination (p.e)")
    plt.ylabel("Charge Err %")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return

#FlasherChargeResolutionNSBasErrTrueMeasured()

def FlasherChargeResolutionVaryTimePlotWaveFormWithNSB():
    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)
    # for n_events in range(0,100,50):
    n_events = 1

    # readouts=[]
    waveforms = []
    legends = []
    for flasher_time_var in np.arange(1, 100, 20):
        illumination = 1000
        # for illumination in tqdm(np.logspace(1, 3, 10)): # up to 1000 illumination in log space
        for iev in range(n_events):
            flasher = source.get_flasher_illumination(time=flasher_time_var, illumination=illumination,
                                                      flasher_pulse_width=4.5,
                                                      illumination_err=0.2,
                                                      pulse_width_err=0.035)
            # nsb = source.get_nsb(rate=100)
            # pe = flasher + nsb
            pe = flasher
            # Add in NSB

            # pe = source.get_uniform_illumination(time=60, illumination=illumination)
            readout = acquisition.get_continuous_readout(pe)
            waveform = acquisition.get_sampled_waveform(readout)

            # readouts.append(readout)
            waveforms.append(waveform)

            # Charge Extraction
            measured_charge = waveform.sum(1)
            true_charge = pe.get_photoelectrons_per_pixel(n_pixels)
            legends.append(
                "Start " + str(flasher_time_var) + " mc/tc " + str(measured_charge / true_charge) + " mc " + str(
                    measured_charge) + ", tc " + str(true_charge))
    nsb_legends=[]
    nsb_waveforms=[]

    # How big is NSB waveform
    for nsb_rate in [100,200,300,400,500,1000,2000]:
        nsb = source.get_nsb(rate=nsb_rate)
        readout = acquisition.get_continuous_readout(nsb)
        waveform = acquisition.get_sampled_waveform(readout)
        nsb_legends.append("NSB Rate= " + str(nsb_rate) + " MHz")
        nsb_waveforms.append(waveform)

    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude (p.e.)')
    image = CameraImage.from_coordinates(camera.mapping.pixel, ax=ax1)
    image.add_colorbar("Amplitude (p.e.)")
    image.image = waveform.max(1) / camera.photoelectron_pulse.height
    for waveform, legend in zip(waveforms, legends):
        ax2.plot(waveform.T, label=legend)

    for waveform, legend in zip(nsb_waveforms, nsb_legends):
        ax2.plot(waveform.T, label=legend)

    ax2.legend()
    plt.show()

    return

#print ("Running FlasherChargeResolutionVaryTimePlotWaveFormWithNSB")
#FlasherChargeResolutionVaryTimePlotWaveFormWithNSB()


# 12/6/21 Vary NSB at 2 illuminations and determine fractional error (no pedestal subtraction)
# Measured charge is whole waveform rather than by using charge extractor to get 50 ns around peak so this method is deprecated.
def Flash_100_200_pe_nsb_no_correction():

    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)
    n_events=1000
    #for n_events in range(0,100,50):

    # Just flashers no NSB
    #
    nsbs=[]
    illuminations=[]
    fractional_errors=[]
    fractional_errors_avg=[]
    for nsb_MHz in [50, 100, 200, 300,1000]:
        for illumination in [100,200]:
            MeasuredChargeSum = 0.0
            TrueChargeSum = 0.0
            event_counter=1
            for iev in range(n_events):
                flasher = source.get_flasher_illumination(time=50, illumination=illumination,
                                                          flasher_pulse_width=4.5,
                                                          illumination_err=0.2,
                                                          pulse_width_err=0.035)

                true_charge = flasher.get_photoelectrons_per_pixel(n_pixels)



                # Add in NSB
                nsb = source.get_nsb(rate=nsb_MHz)
                pe = flasher + nsb



                readout = acquisition.get_continuous_readout(pe)
                waveform = acquisition.get_sampled_waveform(readout)

                # Charge Extraction
                measured_charge = waveform.sum(1)
                MeasuredChargeSum += measured_charge
                TrueChargeSum += true_charge

                fractional_error=((measured_charge-true_charge)/true_charge)
                nsbs.append(nsb_MHz)
                illuminations.append(illumination)
                fractional_errors.append(fractional_error)

                avg_measured_charge=MeasuredChargeSum / event_counter # moving average
                avg_true_charge=TrueChargeSum / event_counter

                fractional_errors_avg.append((avg_measured_charge-avg_true_charge)/avg_true_charge)
                event_counter+=1
    # iterate and plot
    for nsb_MHz in [50, 100, 200, 300, 1000]:
        for illumination in [100, 200]:
            plot_fractional_errors=[]
            plot_event_numbers=[]
            plot_event_number=0
            plot_avg_fractional_errors=[]
            for sim_fractional_error, sim_nsb,sim_illumination in zip(fractional_errors, nsbs, illuminations ):
                if sim_nsb==nsb_MHz and sim_illumination==illumination:
                    plot_event_number+=1
                    plot_event_numbers.append(plot_event_number)
                    plot_fractional_errors.append(sim_fractional_error)
            plt.plot(plot_event_numbers, plot_fractional_errors, '-', label="Illumination " + str(illumination) + " p.e.  NSB " + str(nsb_MHz) + " MHz ")

    plt.legend(loc="best")
    plt.xlabel("Event Number ")
    plt.ylabel("Fractional Error (MC-TC)/TC")

    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.clf()
    for illumination in [100, 200]:
        for nsb_MHz in [50, 100, 200, 300, 1000]:

            plot_event_numbers=[]
            plot_event_number=0
            plot_avg_fractional_errors=[]
            for sim_fractional_error, sim_nsb,sim_illumination in zip(fractional_errors_avg, nsbs, illuminations ):
                if sim_nsb==nsb_MHz and sim_illumination==illumination:
                    plot_event_number+=1
                    plot_event_numbers.append(plot_event_number)
                    plot_avg_fractional_errors.append(sim_fractional_error)
            plt.plot(plot_event_numbers, plot_avg_fractional_errors, '-', label="Illumination " + str(illumination) + " p.e.  NSB " + str(nsb_MHz) + " MHz ")

    plt.legend(loc="best")
    plt.xlabel("Event Number ")
    plt.ylabel("Fractional Error - Running MC and TC Avg - (MC-TC)/TC")

    plt.xscale('log')
    plt.yscale('log')
    plt.show()

    return

#Deprecated
#print("Running Flash_100_200_pe_nsb_no_correction")
#Flash_100_200_pe_nsb_no_correction()



# 100 and 200 p.e. - no AC coupling - 'pedestal' subtraction - NSB constant from a range of values
# 14/6/21 based on subtract average of 20 NSB samples
#

def Flash_100_200_pe_subtract_nsb_pedestal():

    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)
    n_events=1000

    # Get average 20 NSB samples for subtraction
    #
    avg_nsbs=[]
    for nsb_MHz in [50, 100, 200, 300, 1000]:
        measured_charge_sum=0.0
        for x in range(20):
            nsb = source.get_nsb(rate=nsb_MHz)
            readout = acquisition.get_continuous_readout(nsb)
            waveform = acquisition.get_sampled_waveform(readout)

            # Charge Extraction
            measured_charge = waveform.sum(1)
            measured_charge_sum+=measured_charge
        avg_nsbs.append((nsb_MHz,measured_charge_sum/20))


    # nsb + flashers
    nsbs=[]
    illuminations=[]
    fractional_errors=[]
    fractional_errors_avg=[]
    for nsb_MHz in [50, 100, 200, 300,1000]:
        for illumination in [100,200]:
            MeasuredChargeSum = 0.0
            TrueChargeSum = 0.0
            event_counter = 1
            for iev in range(n_events):
                flasher = source.get_flasher_illumination(time=50, illumination=illumination,
                                                          flasher_pulse_width=4.5,
                                                          illumination_err=0.2,
                                                          pulse_width_err=0.035)

                true_charge = flasher.get_photoelectrons_per_pixel(n_pixels)



                # Add in NSB
                nsb = source.get_nsb(rate=nsb_MHz)
                pe = flasher + nsb



                readout = acquisition.get_continuous_readout(pe)
                waveform = acquisition.get_sampled_waveform(readout)

                # Charge Extraction
                measured_charge = waveform.sum(1)

                # correct measued charge for NSB

                # subtract nsb from average measured charge
                for j in avg_nsbs:
                    if j[0] == nsb_MHz:
                        measured_charge=measured_charge-j[1]
                        break
                MeasuredChargeSum += measured_charge
                TrueChargeSum += true_charge

                fractional_error=((measured_charge-true_charge)/true_charge)
                nsbs.append(nsb_MHz)
                illuminations.append(illumination)
                fractional_errors.append(fractional_error)

                avg_measured_charge=MeasuredChargeSum / event_counter
                avg_true_charge=TrueChargeSum / event_counter

                fractional_errors_avg.append((avg_measured_charge-avg_true_charge)/avg_true_charge)
                event_counter+=1
    # iterate and plot
    for illumination in [100]:
        for nsb_MHz in [50, 100, 200, 300, 1000]:

            plot_event_numbers=[]
            plot_event_number=0
            plot_avg_fractional_errors=[]
            for sim_fractional_error, sim_nsb,sim_illumination in zip(fractional_errors_avg, nsbs, illuminations ):
                if sim_nsb==nsb_MHz and sim_illumination==illumination:
                    plot_event_number+=1
                    plot_event_numbers.append(plot_event_number)
                    plot_avg_fractional_errors.append(sim_fractional_error)
            plt.plot(plot_event_numbers, plot_avg_fractional_errors, '-', label="Illumination " + str(illumination) + " p.e.  NSB " + str(nsb_MHz) + " MHz ")

    plt.legend(loc="best")
    plt.xlabel("Event Number ")
    plt.ylabel("Fractional Error - Running MC and TC Avg - (MC-TC)/TC")

    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

    for illumination in [200]:
        for nsb_MHz in [50, 100, 200, 300, 1000]:

            plot_event_numbers=[]
            plot_event_number=0
            plot_avg_fractional_errors=[]
            for sim_fractional_error, sim_nsb,sim_illumination in zip(fractional_errors_avg, nsbs, illuminations ):
                if sim_nsb==nsb_MHz and sim_illumination==illumination:
                    plot_event_number+=1
                    plot_event_numbers.append(plot_event_number)
                    plot_avg_fractional_errors.append(sim_fractional_error)
            plt.plot(plot_event_numbers, plot_avg_fractional_errors, '-', label="Illumination " + str(illumination) + " p.e.  NSB " + str(nsb_MHz) + " MHz ")

    plt.legend(loc="best")
    plt.xlabel("Event Number ")
    plt.ylabel("Fractional Error - Running MC and TC Avg - (MC-TC)/TC")

    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()


    return

#print ("Running Flash_100_200_pe_subtract_nsb_pedestal")
#Flash_100_200_pe_subtract_nsb_pedestal()

#26 June 2021 - Subtract pedestal but do 200, 500, 1000 runs each of 100 flasher events (moving avg) with 20 NSB samples subtracted (constant NSB rate)
# Print Mean and SEM of mean for multiple runs
def Flash_100_200_pe_subtract_nsb_pedestal_iterate():

    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)
    #n_events=1000
    n_events=100
    # Get average 20 NSB samples for subtraction
    #
    avg_nsbs=[]
    for nsb_MHz in [50, 100, 200, 300, 1000]:
        measured_charge_sum=0.0
        for x in range(20):
            nsb = source.get_nsb(rate=nsb_MHz)
            readout = acquisition.get_continuous_readout(nsb)
            waveform = acquisition.get_sampled_waveform(readout)

            # Charge Extraction
            measured_charge = waveform.sum(1)
            measured_charge_sum+=measured_charge
        avg_nsbs.append((nsb_MHz,measured_charge_sum/20))


    # nsb + flashers
    nsbs=[]
    illuminations=[]
    fractional_errors=[]
    fractional_errors_avg=[]
    for iterate_event_run_count in [200,500,1000]:
        for illumination in [100, 200]:
            for nsb_MHz in [50, 100, 200, 300,1000]:

                fractional_errors_n_runs=[]
                for iterate_events in range(iterate_event_run_count):
                    MeasuredChargeSum = 0.0
                    TrueChargeSum = 0.0
                    event_counter = 1
                    MCs=[]
                    TCs=[]
                    for iev in range(n_events):
                        flasher = source.get_flasher_illumination(time=50, illumination=illumination,
                                                                  flasher_pulse_width=4.5,
                                                                  illumination_err=0.2,
                                                                  pulse_width_err=0.035)

                        true_charge = flasher.get_photoelectrons_per_pixel(n_pixels)



                        # Add in NSB
                        nsb = source.get_nsb(rate=nsb_MHz)
                        pe = flasher + nsb



                        readout = acquisition.get_continuous_readout(pe)
                        waveform = acquisition.get_sampled_waveform(readout)

                        # Charge Extraction
                        measured_charge = waveform.sum(1)

                        # correct measued charge for NSB

                        # subtract nsb from average measured charge
                        for j in avg_nsbs:
                            if j[0] == nsb_MHz:
                                measured_charge=measured_charge-j[1]
                                break
                        MeasuredChargeSum += measured_charge
                        TrueChargeSum += true_charge
                        MCs.append(measured_charge)
                        TCs.append(true_charge)
                        fractional_error=((measured_charge-true_charge)/true_charge)
                        nsbs.append(nsb_MHz)
                        illuminations.append(illumination)
                        fractional_errors.append(fractional_error)

                        avg_measured_charge=MeasuredChargeSum / event_counter # running avg
                        avg_true_charge=TrueChargeSum / event_counter # running avg
                        fract_error=(avg_measured_charge-avg_true_charge)/avg_true_charge # fractional err determined on running average
                        fractional_errors_avg.append(fract_error)
                        # print ("-- consistency check ")
                        # print (str(fract_error))
                        # print (str(np.mean(fractional_errors,dtype=np.float64)))
                        # print (str((np.mean(MCs)-np.mean(TCs))/np.mean(TCs)))
                        # print ("--")
                        event_counter+=1
                    fractional_errors_n_runs.append(fract_error)
                print(str(iterate_event_run_count) + " runs, Illumination," + str(illumination) + ",NSB," + str(nsb_MHz) + ",MEAN," + str(np.mean(fractional_errors_n_runs)) +  ",SEM," + str(np.std(fractional_errors_n_runs)/np.sqrt(iterate_event_run_count)))
    # iterate and plot
    for illumination in [100]:
        for nsb_MHz in [50, 100, 200, 300, 1000]:

            plot_event_numbers=[]
            plot_event_number=0
            plot_avg_fractional_errors=[]
            for sim_fractional_error, sim_nsb,sim_illumination in zip(fractional_errors_avg, nsbs, illuminations ):
                if sim_nsb==nsb_MHz and sim_illumination==illumination:
                    plot_event_number+=1
                    plot_event_numbers.append(plot_event_number)
                    plot_avg_fractional_errors.append(sim_fractional_error)

    for illumination in [200]:
        for nsb_MHz in [50, 100, 200, 300, 1000]:

            plot_event_numbers=[]
            plot_event_number=0
            plot_avg_fractional_errors=[]
            for sim_fractional_error, sim_nsb,sim_illumination in zip(fractional_errors_avg, nsbs, illuminations ):
                if sim_nsb==nsb_MHz and sim_illumination==illumination:
                    plot_event_number+=1
                    plot_event_numbers.append(plot_event_number)
                    plot_avg_fractional_errors.append(sim_fractional_error)
    return
#print("Flash_100_200_pe_subtract_nsb_pedestal_iterate")
# Flash_100_200_pe_subtract_nsb_pedestal_iterate()


# Deprecated - use common routine performance.pedestal.obtain_pedestal(camera,extractor,nsb_MHz) instead
def get_nsb_pedestal(nsb_MHz, no_of_nsb_obs):
    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)

    nsb_charges=[]
    for x in range(no_of_nsb_obs):
        nsb = source.get_nsb(rate=nsb_MHz)
        readout = acquisition.get_continuous_readout(nsb)
        waveform = acquisition.get_sampled_waveform(readout)

        # Charge Extraction
        measured_charge = waveform.sum(1)
        nsb_charges.append(measured_charge)

    avg_nsb_charge=np.mean(nsb_charges)
    return avg_nsb_charge

def print_nsb_charge():
    for no_of_nsb_obs in [5,20,100]:
        for nsb_MHz in [50, 100, 200, 300, 1000]:
            print ("NSB obs, " + str(no_of_nsb_obs)  + " ,NSB rate, " + str(nsb_MHz) + " ,NSB charge, " + str(get_nsb_pedestal(nsb_MHz, no_of_nsb_obs)))
    return

#print_nsb_charge()

# uses common peak extraction and pedestal correction
def get_fractional_err_flashers(nsb_MHz,nsb_pedestal_charge,illumination,flasher_pulse_width,illumination_err, pulse_width_err):

    # Define the camera
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=1),  # Only need a single pixel
        continuous_readout_duration=500,  # Only need a single-waveform's worth of readout
        n_waveform_samples=500,
    )
    n_pixels = camera.mapping.n_pixels
    pixel_array = np.arange(n_pixels)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)
    charge_resolution = ChargeResolution(mc_true=True)

    # assume delayed start in readout at 50 ns till we get wrap around waveform issue sorted
    #
    flasher = source.get_flasher_illumination(time=50, illumination=illumination,
                                              flasher_pulse_width=flasher_pulse_width,
                                              illumination_err=illumination_err,
                                              pulse_width_err=pulse_width_err)

    extractor = performance.ChargeExtractor.from_camera(camera)

    #true_charge = flasher.get_photoelectrons_per_pixel(n_pixels) # old way
    readout = acquisition.get_continuous_readout(flasher)
    waveform = acquisition.get_sampled_waveform(readout)
    true_charge = extractor.extract(waveform, 50)

    # use peak extraction !

    # Add in NSB
    nsb = source.get_nsb(rate=nsb_MHz)
    pe = flasher + nsb

    readout = acquisition.get_continuous_readout(pe)
    waveform = acquisition.get_sampled_waveform(readout)

    # Charge Extraction - old method - integrate whole waveform - used in plots prior to 31st July 2021
    # measured_charge = waveform.sum(1)

    # Charge Extraction New method - integrate around the peak +/- 10 ns
    #
    measured_charge=extractor.extract(waveform,50)
    measured_charge=measured_charge-nsb_pedestal_charge

    fractional_error=(measured_charge-true_charge)/true_charge

    return fractional_error,measured_charge,true_charge


def get_fractional_err_flashers_moving_avg(no_of_flashes,nsb_MHz,nsb_pedestal_charge,flasher_illumination,pulse_width,illumination_err, pulse_width_err):
    event_nos=[]
    fractional_err_charge_moving_avgs=[]
    measured_charges=[]
    true_charges=[]
    for x in range(no_of_flashes):
        fractional_err, measured_charge, true_charge = get_fractional_err_flashers(nsb_MHz, nsb_pedestal_charge, flasher_illumination, pulse_width, illumination_err, pulse_width_err)
        measured_charges.append(measured_charge)
        true_charges.append(true_charge)
        event_nos.append(x)
        fractional_err_charge_moving_avgs.append((np.mean(measured_charges)-np.mean(true_charges))/np.mean(true_charges))
    return event_nos,fractional_err_charge_moving_avgs


# 26/6/2021
flasher_illuminations=[100,200] # range of illuminations to try in p.e.
nsb_rates=[50, 100, 200, 300, 1000] # range of rates to try in MHz
pulse_width=4.5 # flasher pulse width ns
illumination_err=0.2 # flasher illumination error
pulse_width_err=0.035 # flasher pulse width error
no_of_nsb_obs=20 # How many NSB samples to take, take average of these and subtract as a pedestal
no_of_flashes=500 # Number of flashes for PlotFractionalError
no_of_consistent_runs=10 # No of CONSECUTIVE runs over which we must achieve target error
target_fractional_err=0.02 # target fractional error , defined as (measured-true)/true
no_of_flashes_step=1 # reach goal quicker and accept some imprecision in number of flashes - this actually isnt't so usefull - increasing step size doesn't seem to help much

def PlotFractionalErr(no_of_nsb_obs,no_of_flashes, flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err):
    extractor = performance.ChargeExtractor.from_camera(camera)
    for flasher_illumination in flasher_illuminations:
        print("Flasher Illumination " + str(flasher_illumination) + " p.e. ")
        for nsb_MHz in nsb_rates:
            print("NSB  " + str(nsb_MHz) + " MHz ")
            nsb_pedestal_charge_sum=0
            for x in range(no_of_nsb_obs):
                nsb_pedestal_charge_sum += performance.pedestal.obtain_pedestal(camera, extractor, nsb_MHz)
            nsb_pedestal_charge=nsb_pedestal_charge_sum/no_of_nsb_obs
            fractional_err_charge_moving_avgs=[]
            event_nos=[]

            event_nos,fractional_err_charge_moving_avgs=get_fractional_err_flashers_moving_avg(no_of_flashes, nsb_MHz, nsb_pedestal_charge, flasher_illumination,
                                                   pulse_width, illumination_err, pulse_width_err)

            plt.plot(event_nos, fractional_err_charge_moving_avgs, '-', label="Illumination " + str(flasher_illumination) + " p.e.  NSB " + str(nsb_MHz) + " MHz ")
    plt.legend(loc="best")
    plt.xlabel("Event Number ")
    plt.ylabel("Fractional Error - Running MC and TC Avg - (MC-TC)/TC")
    plt.show()

    return

#flasher_illuminations=[100]
#print ("PlotFractionalErr 100 pe")
#PlotFractionalErr(no_of_nsb_obs,no_of_flashes, flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err)

#flasher_illuminations=[200]
#print ("PlotFractionalErr 200 pe")
#PlotFractionalErr(no_of_nsb_obs,no_of_flashes, flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err)


# no_of_runs - how many callibrations to attempt - we should achieve target_fractional_err consistently in this consecutive number of calibration runs
def GetNumberOfFlashesToReachTargetFractionalError(no_of_flashes_step,no_of_consistent_runs,target_fractional_err,no_of_nsb_obs,flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err):
    print ("Begin GetNumberOfFlashesToReachTargetFractionalError - no of NSB obs " + str(no_of_nsb_obs))
    extractor = performance.ChargeExtractor.from_camera(camera)
    for flasher_illumination in flasher_illuminations:
        #print("Flasher Illumination " + str(flasher_illumination) + " p.e. ")
        for nsb_MHz in nsb_rates:
            fractional_errors_n_runs = []
            achieved_target_fractional_err=False
            no_of_flashes=0
            while (achieved_target_fractional_err==False):
                no_of_flashes+=no_of_flashes_step
                print ("no_of_flashes " + str(no_of_flashes) + " NSB " + str(nsb_MHz))
                debug_fract_errs=[]

                for x in range (no_of_consistent_runs): # we need to meet the target err consistently in this number of runs
                    nsb_pedestal_charge_sum = 0
                    for x in range(no_of_nsb_obs):
                        nsb_pedestal_charge_sum += performance.pedestal.obtain_pedestal(camera, extractor, nsb_MHz)
                    nsb_pedestal_charge = nsb_pedestal_charge_sum / no_of_nsb_obs
                    #nsb_pedestal_charge=performance.pedestal.obtain_pedestal(camera, extractor, nsb_MHz)
                    fractional_err_charge_moving_avgs=[]
                    event_nos=[]
                    event_nos,fractional_err_charge_moving_avgs=get_fractional_err_flashers_moving_avg(no_of_flashes, nsb_MHz, nsb_pedestal_charge, flasher_illumination,
                                                           pulse_width, illumination_err, pulse_width_err)
                    debug_fract_errs.append(fractional_err_charge_moving_avgs[-1])


                    if (abs(fractional_err_charge_moving_avgs[-1]) > target_fractional_err): # get the last fractional error after no_of_flashes
                        # this is the failure case so don't bother to iterate further
                        #print ("Fractional error, " + str(fractional_err_charge_moving_avgs[-1]) + " ,Breach at run " + str(x)  + " with " + str(no_of_flashes) + " flashes ")
                        break
                if (abs(fractional_err_charge_moving_avgs[-1]) > target_fractional_err):
                    continue # attempt to get target fractional error again which must be acheived in no_of_consistent_runs iterations
                else:
                    #print ("run " + str(x))
                    achieved_target_fractional_err = True# stopping case
            output_file="/home/sheridan/Documents/SSTCAM/RESEARCH_NOTE/GetNoOfFlashes_for_FE_" + str (target_fractional_err)
            with open(output_file, 'a') as f:
                print("Illumination," + str(flasher_illumination) + ",NSB," + str(nsb_MHz) + ",Target Fractional error," + str(target_fractional_err) + " achieved for " + str(no_of_consistent_runs) + " runs in " + str(no_of_flashes) + " flashes", file=f)
            f.close()
            print("Illumination," + str(flasher_illumination) + ",NSB," + str(nsb_MHz) + ",Target Fractional error," + str(target_fractional_err) + " achieved for " + str(no_of_consistent_runs) + " runs in " + str(no_of_flashes) + " flashes")
    return
print ("GetNumberOfFlashesToReachTargetFractionalError")
for target_fractional_err in [0.01, 0.005]:
#for target_fractional_err in [0.009,0.008,0.007,0.006,0.005]:
    GetNumberOfFlashesToReachTargetFractionalError(no_of_flashes_step,no_of_consistent_runs,target_fractional_err,no_of_nsb_obs,flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err)

# 30th July 2021 - simulate the drop in illumination due to over-voltage drop for nsb of a given rate in MHz, illumination is p.e.
# Define constants for run that follows
flasher_illuminations=[100,200] # range of illuminations to try in p.e.
pulse_width=4.5 # flasher pulse width ns
illumination_err=0.2 # flasher illumination error
pulse_width_err=0.035 # flasher pulse width error
no_of_nsb_obs=20 # How many NSB samples to take, take average of these and subtract as a pedestal

def get_illumination_for_nsb(illumination,nsb_MHz):
    percentage_drop_per_MHz=0.016 # from Jon Lapington Laser Tests of simulated NSB effect on a laser pulse amplitude
    illumination=illumination-(illumination * (nsb_MHz * percentage_drop_per_MHz / 100))
    return illumination

# Simulate linear NSB at a given time in seconds. Initial_nsb_rate is nsb at time zero. All rates in MHz.
def get_linear_nsb_rate(time_s, initial_nsb_MHz, nsb_MHz_rate_of_change_per_s):
    nsb_MHz = initial_nsb_MHz + (nsb_MHz_rate_of_change_per_s*time_s)
    return nsb_MHz

# time_s - Time in seconds that we want to extract NSB for - this is unbounded obviously
# min_nsb_MHz - lower bound of NSB range
# max_nsb_MHz - upper bound of NSB range
# rise_time_s - Time in seconds to go from min NSB to max NSB and vice versa

def get_sin_nsb_rate(time_s,min_nsb_MHz, max_nsb_MHz,rise_time_s):
    nsb_MHz=min_nsb_MHz+(0.5*(max_nsb_MHz-min_nsb_MHz))*(np.sin(1.5*np.pi+np.pi/rise_time_s*time_s)+1) # repeats indefinitely , wavelength 2PI start at min value

    return nsb_MHz

# test get nsb sin rate
# time_s=0
# min_nsb_MHz=100
# max_nsb_MHz=200
# rise_time_s=50
# for time_s in range(0,200,1):
#     print (str(time_s) + " " + str(get_sin_nsb_rate(time_s,min_nsb_MHz, max_nsb_MHz,rise_time_s)))
#     get_sin_nsb_rate(time_s, min_nsb_MHz, max_nsb_MHz, rise_time_s)


# Measured charge is nsb pedestal corrected and includes nsb charge at given rate
# Allow for NSB rate of change
# readout_time_s - Time in run at which flasher calibration starts
# no_of_flashes - this is within a single calibration run - there will be multiple calibration runs abritarily spaced in time
# flasher_rate_Hz - 5-20 typically
# nsb_MHz_at_time_zero - NSB at start of observing - we will adjust this depending on readout_time_s and nsb rate of change - linear increase
# nsb_rate_of_change_MHz_s - NSB rate of change per second
# nsb_pedestal_charge - This any pedestal we want - try first with pedestal at time zero
# flasher_illumination_time_zero - we reduce this depending on time and NSB to simulate overvoltage drop with NSB
def get_flasher_measured_and_true_linear_NSB(readout_time_s, no_of_flashes, flasher_rate_Hz,nsb_MHz_at_time_zero, nsb_rate_of_change_MHz_s,nsb_pedestal_charge,flasher_illumination_time_zero,pulse_width,illumination_err, pulse_width_err):

    measured_charges=[]
    true_charges=[]
    nsbs=[]
    time_between_flashes_s=1/flasher_rate_Hz

    for x in range(no_of_flashes): # allow also for change in NSB while flasher calibration going on - probably not a significant effect but nice to include for star in pixel sim
        nsb_MHz=get_linear_nsb_rate(readout_time_s,nsb_MHz_at_time_zero,nsb_rate_of_change_MHz_s) # NSB is changing even when calibration run
        nsbs.append(nsb_MHz)
        flasher_illumination=get_illumination_for_nsb(flasher_illumination_time_zero,nsb_MHz) # The true initial flasher illumination is changing depending on this changed NSB
        true_charges.append(flasher_illumination)
        fractional_err, measured_charge, true_charge = get_fractional_err_flashers(nsb_MHz, nsb_pedestal_charge, flasher_illumination, pulse_width, illumination_err, pulse_width_err)
        measured_charges.append(measured_charge) # pedestal corrected includes nsb
        readout_time_s+=time_between_flashes_s # everything is driven by time offset from a zero point at start of  observations

    return np.mean(measured_charges), np.mean(true_charges), np.mean(nsbs)


# Measured charge is nsb pedestal corrected and includes nsb charge at given rate
# Allow for NSB rate of change
# readout_time_s - Time in run at which flasher calibration starts
# no_of_flashes - this is within a single calibration run - there will be multiple calibration runs abritarily spaced in time
# flasher_rate_Hz - 5-20 typically
# nsb_pedestal_charge - This any pedestal we want - try first with pedestal at time zero
# flasher_illumination_time_zero - we reduce this depending on time and NSB to simulate overvoltage drop with NSB
# min_nsb_MHz - Minimum NSB (also the inital NSB at time zero)
# max_nsb_MHz - Maximum NSB
# rise_time_s - Time to go from min to max NSB and down again
#
def get_flasher_measured_and_true_sine_NSB(readout_time_s, no_of_flashes, flasher_rate_Hz, nsb_pedestal_charge,flasher_illumination_time_zero,pulse_width,illumination_err, pulse_width_err,min_nsb_MHz, max_nsb_MHz,rise_time_s):

    measured_charges=[]
    true_charges=[]
    nsbs=[]
    time_between_flashes_s=1/flasher_rate_Hz

    for x in range(no_of_flashes): # allow also for change in NSB while flasher calibration going on - probably not a significant effect but nice to include for star in pixel sim
        nsb_MHz=get_sin_nsb_rate(readout_time_s,min_nsb_MHz, max_nsb_MHz,rise_time_s) # NSB is changing even when calibration run
        nsbs.append(nsb_MHz)
        flasher_illumination=get_illumination_for_nsb(flasher_illumination_time_zero,nsb_MHz) # The true initial flasher illumination is changing depending on this changed NSB
        true_charges.append(flasher_illumination)
        fractional_err, measured_charge, true_charge = get_fractional_err_flashers(nsb_MHz, nsb_pedestal_charge, flasher_illumination, pulse_width, illumination_err, pulse_width_err)
        measured_charges.append(measured_charge) # pedestal corrected includes nsb
        readout_time_s+=time_between_flashes_s # everything is driven by time offset from a zero point at start of  observations

    return np.mean(measured_charges), np.mean(true_charges), np.mean(nsbs)


# flasher_rate_Hz - typically 5-20
#
def get_calibration_charge_linear_NSB(take_multiple_pedestal,nsb_MHz_at_time_zero,flasher_illumination_time_zero,nsb_rate_of_change_MHz_s,flasher_rate_Hz,no_of_flashes_in_calibration_step,interval_between_calibration_s):

    # Goal - what is effect of flasher rate, number of flashes and varying NSB rate on determining calibration coefficient and miscalibration factor (defined below)
    # Overall steps - a flasher calibration run can have a varying number of flashes and rate:
    # 0 - take_multiple_pedestal = False Take one initial pedestal as the start NSB t=0, take_multiple_pedestal= True - Take NSB pedestal start of each flash run
    # 1 - Get original pedestal corrected (OPC) charge for flasher calibration run at t=0 ( assume no variation NSB here )
    # 2 - Measure reduced pedestal corrected charge (MRPC) at any time t and simulate reduced measured illumination due to increasing NSB at a given rate, calibration coefficient
    #     CC=OPC/RPC
    # 3 - At any NSB we define a true (TPC) charge with no statistical errors, this is the drift of OPC and RPC would be same as OPC if perfect.
    #     TPC-RPC is the miscalibration factor (MF)
    # 4 - Determine mean, standard deviation and root mean square error on the CC and MCF

    extractor=performance.ChargeExtractor.from_camera(camera)
    nsb_pedestal_charge = performance.pedestal.obtain_pedestal(camera, extractor, nsb_MHz_at_time_zero) # step 0


    # how many seconds run to get to 1000 MHz NSB ?
    observing_duration_s=1000 / nsb_rate_of_change_MHz_s
    calibration_run_duration_s= no_of_flashes_in_calibration_step / flasher_rate_Hz

    running_time=0
    # place calibration runs roughly into observing duration
    #
    NSBs=[]
    MRPCs=[]
    TPCs=[]
    while running_time < observing_duration_s:
        if (take_multiple_pedestal):
            nsb_MHz_for_frequent_pedestal = get_linear_nsb_rate(running_time, nsb_MHz_at_time_zero, nsb_rate_of_change_MHz_s)
            nsb_pedestal_charge = performance.pedestal.obtain_pedestal(camera, extractor, nsb_MHz_for_frequent_pedestal)

        MRPC,TPC,NSB=get_flasher_measured_and_true_linear_NSB(running_time, no_of_flashes_in_calibration_step, flasher_rate_Hz, \
                               nsb_MHz_at_time_zero,nsb_rate_of_change_MHz_s, nsb_pedestal_charge, flasher_illumination_time_zero, pulse_width, \
                               illumination_err, pulse_width_err) # step 2 - avg reduced pedestal corrected charge with increasing NSB which also allows for NSB change within run

        NSBs.append(NSB)
        MRPCs.append(MRPC) # collect the mean pedestal corrected measured charge from a calibration run ( consisting of many flashes )
        TPCs.append(TPC) # collect the mean true charge which is the reduced flasher illumination to simulate reduced overvolatge as a result of NSB
        #print('Time ' + str(running_time) + ' Measured '+ str(RPC) + ' True ' + str(TPC))

        running_time=running_time + calibration_run_duration_s + interval_between_calibration_s

    return MRPCs, TPCs, NSBs



nsb_MHz_at_time_zero=50
flasher_illumination_time_zero=100
nsb_rate_of_change_MHz_s=0.2
flasher_rate_Hz=10
flasher_rates_Hz=[20]
no_of_flashes_in_calibration_step=10
interval_between_calibration_s=600
no_of_calibration_iterations=10


def get_stats(bin_NSB,all_MPRCs, all_TPCs, all_NSBs,output_file="/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/log.txt"):
    # effectively binning by NSB to identify the individual calibration run and re-bin by this for stats

    MPRC_binned_NSB = []
    TPC_binned_NSB = []

    for MPRC, TPC, NSB in zip(all_MPRCs, all_TPCs, all_NSBs):
        if NSB == bin_NSB:
            print ("NSB, " + str(NSB) + ",MPRC," + str(MPRC) + ",TPC," + str(TPC))
            MPRC_binned_NSB.append(MPRC)
            TPC_binned_NSB.append(TPC)
    residuals = []
    for measured_charge, true_charge in zip(MPRC_binned_NSB, TPC_binned_NSB):
        residuals.append((measured_charge - true_charge) ** 2)
    RMSE = np.sqrt(np.mean(residuals))
    print ("NSB " + str(bin_NSB) + " RMSE " + str(RMSE))
    # Get stats
    with open(output_file, 'a') as f:
        print('NSB ' + str(bin_NSB) + " RMSE " + str(RMSE) + " MPRC mean & std dev " + \
          str(np.mean(MPRC_binned_NSB)) + "," + str(np.std(MPRC_binned_NSB)) + \
          " TPC mean & std dev " + str(np.mean(TPC_binned_NSB)) + "," + \
          str(np.std(TPC_binned_NSB)),file=f )
    f.close()
    return np.mean(MPRC_binned_NSB), np.std(MPRC_binned_NSB), RMSE,np.mean(TPC_binned_NSB)

# This assumes a linear rate of change increasing NSB
#
def run_calibration(take_multiple_pedestal,output_file,no_of_calibration_iterations,nsb_MHz_at_time_zero,flasher_illumination_time_zero,nsb_rate_of_change_MHz_s,flasher_rate_Hz,no_of_flashes_in_calibration_step,interval_between_calibration_s):

    all_MPRCs=[]
    all_TPCs=[]
    all_NSBs=[]
    for x in range(no_of_calibration_iterations):
        MRPCs, TPCs, NSBs=get_calibration_charge_linear_NSB(take_multiple_pedestal,nsb_MHz_at_time_zero,flasher_illumination_time_zero,nsb_rate_of_change_MHz_s,\
                                                     flasher_rate_Hz,no_of_flashes_in_calibration_step,interval_between_calibration_s)
        all_MPRCs+=MRPCs
        all_TPCs+=TPCs
        all_NSBs+=NSBs

    # dervive stats from global lists
    np_all_NSBs=np.array(all_NSBs)

    # Determine means, root mean square error and standard deviations for plot / print
    #print ("Flasher rate Hz " + str(flasher_rate_Hz))

    mean_MPRCs=[]
    std_dev_MPRCs=[]
    RMSEs=[]
    mean_NSBs=[]
    mean_TPCs=[]
    for bin_nsb in np.unique(np_all_NSBs):
        mean_MPRC,std_dev_MPRC,RMSE, mean_TPC=get_stats(bin_nsb, all_MPRCs, all_TPCs, all_NSBs,output_file)
        mean_MPRCs.append(mean_MPRC)
        std_dev_MPRCs.append(std_dev_MPRC)
        RMSEs.append(RMSE)
        mean_NSBs.append(bin_nsb)
        mean_TPCs.append(mean_TPC)
        #print ("Mean MPRC " + str(mean_MPRC))
        #print ("Std dev MPRC " + str(std_dev_MPRC))
        #print ("RMSE " + str(RMSE))

    return mean_MPRCs,std_dev_MPRCs,RMSEs,mean_NSBs, mean_TPCs

MPRCs=[]
std_dev_MPRCs=[]
RMSEs=[]
mean_NSBs=[]
no_of_calibration_iterations=10

def new_plt(plt_title, x_label,y_label):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.title.set_text(plt_title)



    return ax1

def plotnormal(ax1, x_array,y_array,series_label):
    ax1.plot(x_array, y_array, '-', label=series_label)
    ax1.legend(loc="upper right")
    return


def show_plt(ax1):
    plt.show()
    return



def PrintParams(output_file,no_of_calibration_iterations,nsb_MHz_at_time_zero,flasher_illumination_time_zero,nsb_rate_of_change_MHz_s,flasher_rate_Hz,no_of_flashes_in_calibration_step,interval_between_calibration_s):
    with open(output_file, 'a') as f:
        print ("no_of_calibration_iterations      " + str (no_of_calibration_iterations), file=f)
        print ("nsb_MHz_at_time_zero              " + str(nsb_MHz_at_time_zero), file=f)
        print ("flasher_illumination_time_zero    " + str(flasher_illumination_time_zero), file=f)
        print ("nsb_rate_of_change_MHz_s          " + str(nsb_rate_of_change_MHz_s), file=f)
        print ("flasher_rate_Hz                   " + str(flasher_rate_Hz), file=f)
        print ("no_of_flashes_in_calibration_step " + str(no_of_flashes_in_calibration_step), file=f)
        print ("interval_between_calibration_s    " + str(interval_between_calibration_s), file=f)
    f.close()
    return

def PltVaryFlasherRateWithTrueIllumination():

    # Vary Flasher Rate - plot against True illumination which excludes NSB and allows for over-voltage reduction
    ax1=new_plt("Vary Flasher Rate - Measured Illumination vs True Illumination", "True charge (p.e.)", "Measured charge (p.e.)")
    ax2=new_plt("Vary Flasher Rate - RMSE vs True Illumination", "True charge (p.e.)", "RMSE (p.e.)")
    ax3=new_plt("Vary Flasher Rate - Std Dev vs True Illumination", "True charge (p.e.)", "Standard Deviation (p.e.)")
    nsb_MHz_at_time_zero=50
    flasher_illumination_time_zero=100
    nsb_rate_of_change_MHz_s=0.2
    #flasher_rate_Hz
    no_of_flashes_in_calibration_step=100
    interval_between_calibration_s=600
    no_of_calibration_iterations=100
    take_frequent_pedestal=True
    SEDLabel = "VaryFlasherRate_5_20_Hz_Initial_NSB_50MHz_100Flashes_debug"
    output_file="/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/" + SEDLabel + '.log'
    for flasher_rate_Hz in [5,10,15,20]:
        print("*****************")
        print("Flasher rate " + str(flasher_rate_Hz) + " Hz " + " " + str(no_of_calibration_iterations) + " runs")
        # output parameters to a log file
        PrintParams(output_file,no_of_calibration_iterations, nsb_MHz_at_time_zero, flasher_illumination_time_zero,
                    nsb_rate_of_change_MHz_s, flasher_rate_Hz, no_of_flashes_in_calibration_step,
                    interval_between_calibration_s)

        # apply multiple calibrations
        mean_MPRCs,std_dev_MPRCs,RMSEs,mean_NSBs,mean_TPCs=run_calibration(take_frequent_pedestal,output_file, no_of_calibration_iterations,nsb_MHz_at_time_zero,flasher_illumination_time_zero,nsb_rate_of_change_MHz_s,flasher_rate_Hz,no_of_flashes_in_calibration_step,interval_between_calibration_s)
        plotnormal(ax1, mean_TPCs,mean_MPRCs,str(flasher_rate_Hz) + " Hz")
        plotnormal(ax2, mean_TPCs, RMSEs,str(flasher_rate_Hz) + " Hz")
        plotnormal(ax3, mean_TPCs, std_dev_MPRCs, str(flasher_rate_Hz) + " Hz")


    #fname = "/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/" + SEDLabel + '.png'
    #plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait',  format=None,
    #            transparent=False, bbox_inches=None, pad_inches=0.1)

    show_plt(ax1)
    show_plt(ax2)
    show_plt(ax3)
    return

#PltVaryFlasherRateWithTrueIllumination()


def PltVaryNSBRateWithTrueIllumination():

    # Vary Flasher Rate - plot against True illumination which excludes NSB and allows for over-voltage reduction
    ax1=new_plt("Vary NSB rate of change - Measured Illumination vs True Illumination", "True charge (p.e.)", "Measured charge (p.e.)")
    ax2=new_plt("Vary NSB rate of change - RMSE vs True Illumination", "True charge (p.e.)", "RMSE (p.e.)")
    ax3=new_plt("Vary NSB rate of change - Std Dev vs True Illumination", "True charge (p.e.)", "Standard Deviation (p.e.)")

    nsb_MHz_at_time_zero=50
    flasher_illumination_time_zero=100

    #flasher_rate_Hz
    no_of_flashes_in_calibration_step=10
    interval_between_calibration_s=600
    no_of_calibration_iterations=10
    flasher_rate_Hz=10
    take_frequent_pedestal = True
    SEDLabel = "VaryNSB_ROC_Initial_NSB_50MHz_100_Flashes_debug"
    output_file="/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/" + SEDLabel + '.log'

    for nsb_rate_of_change_MHz_s in [0.2,0.4,0.6,0.8]:
        print("*****************")
        print("NSB rate of change " + str(nsb_rate_of_change_MHz_s )+" MHz/s")

        PrintParams(output_file,no_of_calibration_iterations, nsb_MHz_at_time_zero, flasher_illumination_time_zero,
                    nsb_rate_of_change_MHz_s, flasher_rate_Hz, no_of_flashes_in_calibration_step,
                    interval_between_calibration_s)


        mean_MPRCs,std_dev_MPRCs,RMSEs,mean_NSBs,mean_TPCs=run_calibration(take_frequent_pedestal,output_file, no_of_calibration_iterations,nsb_MHz_at_time_zero,flasher_illumination_time_zero,nsb_rate_of_change_MHz_s,flasher_rate_Hz,no_of_flashes_in_calibration_step,interval_between_calibration_s)
        plotnormal(ax1, mean_TPCs,mean_MPRCs,str(nsb_rate_of_change_MHz_s) + " MHz/s")
        plotnormal(ax2, mean_TPCs, RMSEs,str(nsb_rate_of_change_MHz_s) + " MHz/s")
        plotnormal(ax3, mean_TPCs, std_dev_MPRCs, str(nsb_rate_of_change_MHz_s) + " MHz/s")

    #show_plt(ax1)
    fname = "/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/" + SEDLabel + '.png'
    #plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait',  format=None,
    #            transparent=False, bbox_inches=None, pad_inches=0.1)

    show_plt(ax1)
    show_plt(ax2)
    show_plt(ax3)

    return

#PltVaryNSBRateWithTrueIllumination()


def PltVaryNoOfFlashesWithTrueIllumination():

    # Vary Flasher Rate - plot against True illumination which excludes NSB and allows for over-voltage reduction
    ax1=new_plt("Vary Flashes in calibration - Measured Illumination vs True Illumination", "True charge (p.e.)", "Measured charge (p.e.)")
    ax2=new_plt("Vary Flashes in calibration - RMSE vs True Illumination", "True charge (p.e.)", "RMSE (p.e.)")
    ax3=new_plt("Vary Flashes in calibration - Std Dev vs True Illumination", "True charge (p.e.)", "Standard Deviation (p.e.)")

    nsb_MHz_at_time_zero=50
    flasher_illumination_time_zero=100
    nsb_rate_of_change_MHz_s=0.2
    interval_between_calibration_s=600
    no_of_calibration_iterations=100
    flasher_rate_Hz=10
    take_frequent_pedestal = True
    SEDLabel = "VaryNo_Of_Flashes_10_200_Initial_NSB_50MHz_1_debug"
    output_file="/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/" + SEDLabel + '.log'

    #for no_of_flashes_in_calibration_step in [10, 15,20]:
    for no_of_flashes_in_calibration_step in [10,50,100,200]:
        print("*****************")
        print(str(no_of_flashes_in_calibration_step )+" flashes in calibration step")
        PrintParams(output_file,no_of_calibration_iterations, nsb_MHz_at_time_zero, flasher_illumination_time_zero,
                    nsb_rate_of_change_MHz_s, flasher_rate_Hz, no_of_flashes_in_calibration_step,
                    interval_between_calibration_s)


        mean_MPRCs,std_dev_MPRCs,RMSEs,mean_NSBs,mean_TPCs=run_calibration(take_frequent_pedestal,output_file, no_of_calibration_iterations,nsb_MHz_at_time_zero,flasher_illumination_time_zero,nsb_rate_of_change_MHz_s,flasher_rate_Hz,no_of_flashes_in_calibration_step,interval_between_calibration_s)
        plotnormal(ax1, mean_TPCs,mean_MPRCs,str(no_of_flashes_in_calibration_step) + " flashes")
        plotnormal(ax2, mean_TPCs, RMSEs,str(no_of_flashes_in_calibration_step) + " flashes")
        plotnormal(ax3, mean_TPCs, std_dev_MPRCs, str(no_of_flashes_in_calibration_step) + " flashes")

    # show_plt(ax1)
    fname = "/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/" + SEDLabel + '.png'
    #plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait',  format=None,
    #            transparent=False, bbox_inches=None, pad_inches=0.1)

    show_plt(ax1)
    show_plt(ax2)
    show_plt(ax3)

    return

#PltVaryNoOfFlashesWithTrueIllumination()


def get_calibration_sine_wave_NSB(take_multiple_pedestal,\
                                  flasher_illumination_time_zero,\
                                  flasher_rate_Hz,\
                                  no_of_flashes_in_calibration_step,\
                                  interval_between_calibration_s,\
                                  no_of_calibration_steps,
                                  min_nsb_MHz, \
                                  max_nsb_MHz,\
                                  rise_time_s):

    # Goal - what is effect of flasher rate, number of flashes and varying NSB rate on determining calibration coefficient and miscalibration factor (defined below)
    # Overall steps - a flasher calibration run can have a varying number of flashes and rate:
    # 0 - take_multiple_pedestal = False Take one initial pedestal as the start NSB t=0, take_multiple_pedestal= True - Take NSB pedestal start of each flash run
    # 1 - Get original pedestal corrected (OPC) charge for flasher calibration run at t=0 ( assume no variation NSB here )
    # 2 - Measure reduced pedestal corrected charge (MRPC) at any time t and simulate reduced measured illumination due to increasing NSB at a given rate, calibration coefficient
    #     CC=OPC/RPC
    # 3 - At any NSB we define a true  charge (TPC) with no statistical errors, this is the drift of OPC and RPC would be same as OPC if perfect.
    #     TPC-RPC is the miscalibration factor (MF)
    # 4 - Determine mean, standard deviation and root mean square error on the CC and MCF

    # Always take at least one pedestal at the start
    extractor=performance.ChargeExtractor.from_camera(camera)
    nsb_pedestal_charge = performance.pedestal.obtain_pedestal(camera, extractor, min_nsb_MHz) # step 0

    running_time=0
    # place calibration runs roughly into observing duration
    #
    NSBs=[]
    MRPCs=[]
    TPCs=[]
    calibration_run_duration_s = no_of_flashes_in_calibration_step / flasher_rate_Hz

    for x in range(no_of_calibration_steps):
        if (take_multiple_pedestal):
            nsb_MHz_for_frequent_pedestal = get_sin_nsb_rate(running_time,min_nsb_MHz, max_nsb_MHz,rise_time_s)
            nsb_pedestal_charge = performance.pedestal.obtain_pedestal(camera, extractor, nsb_MHz_for_frequent_pedestal)

        MRPC,TPC,NSB=get_flasher_measured_and_true_sine_NSB(running_time, no_of_flashes_in_calibration_step, flasher_rate_Hz, \
                               nsb_pedestal_charge, flasher_illumination_time_zero, pulse_width, \
                               illumination_err, pulse_width_err,min_nsb_MHz, max_nsb_MHz, rise_time_s) # step 2 - avg reduced pedestal corrected charge with increasing NSB which also allows for NSB change within run

        NSBs.append(NSB)
        MRPCs.append(MRPC) # collect the mean pedestal corrected measured charge from a calibration run ( consisting of many flashes )
        TPCs.append(TPC) # collect the mean true charge which is the reduced flasher illumination to simulate reduced overvolatge as a result of NSB
        #print('Time ' + str(running_time) + ' Measured '+ str(RPC) + ' True ' + str(TPC))

        running_time=running_time + calibration_run_duration_s + interval_between_calibration_s

    return MRPCs, TPCs, NSBs

def run_calibration_sine_wave_NSB(take_multiple_pedestal,output_file,no_of_calibration_iterations,\
                                  flasher_illumination_time_zero,flasher_rate_Hz,no_of_calibration_steps,\
                                  no_of_flashes_in_calibration_step,interval_between_calibration_s,\
                                  min_nsb_MHz, max_nsb_MHz, rise_time_s):

    all_MPRCs=[]
    all_TPCs=[]
    all_NSBs=[]
    for x in range(no_of_calibration_iterations): # independent calibrations - used to take an overall mean
        MRPCs, TPCs, NSBs=get_calibration_sine_wave_NSB(take_multiple_pedestal,flasher_illumination_time_zero,\
                                                        flasher_rate_Hz,no_of_flashes_in_calibration_step,interval_between_calibration_s,\
                                                        no_of_calibration_steps, min_nsb_MHz, max_nsb_MHz, rise_time_s)
        all_MPRCs+=MRPCs
        all_TPCs+=TPCs
        all_NSBs+=NSBs

    # dervive stats from global lists
    np_all_NSBs=np.array(all_NSBs)

    # Determine means, root mean square error and standard deviations for plot / print
    #print ("Flasher rate Hz " + str(flasher_rate_Hz))

    mean_MPRCs=[]
    std_dev_MPRCs=[]
    RMSEs=[]
    mean_NSBs=[]
    mean_TPCs=[]
    for bin_nsb in np.unique(np_all_NSBs):
        mean_MPRC,std_dev_MPRC,RMSE, mean_TPC=get_stats(bin_nsb, all_MPRCs, all_TPCs, all_NSBs,output_file)
        mean_MPRCs.append(mean_MPRC)
        std_dev_MPRCs.append(std_dev_MPRC)
        RMSEs.append(RMSE)
        mean_NSBs.append(bin_nsb)
        mean_TPCs.append(mean_TPC)
        #print ("Mean MPRC " + str(mean_MPRC))
        #print ("Std dev MPRC " + str(std_dev_MPRC))
        #print ("RMSE " + str(RMSE))

    return mean_MPRCs,std_dev_MPRCs,RMSEs,mean_NSBs, mean_TPCs


def PltVaryTimeBetweenFlashes():
    # Vary Flasher Rate - plot against True illumination which excludes NSB and allows for over-voltage reduction
    ax1=new_plt("Vary Interval between each flasher calibration step  - Measured Illumination vs True Illumination", "True charge (p.e.)", "Measured charge (p.e.)")
    ax2=new_plt("Vary Interval between each flasher calibration step  - RMSE vs True Illumination", "True charge (p.e.)", "RMSE (p.e.)")
    ax3=new_plt("Vary Interval between each flasher calibration step  - Std Dev vs True Illumination", "True charge (p.e.)", "Standard Deviation (p.e.)")

    # A calibration "run" has multiple "steps" with each step being a certain number of flashes at a given frequency
    # We do multiple indepedent runs ( i.e. no_of_calibration_iterations) for the purpose of taking a mean

    flasher_illumination_time_zero=100
    no_of_calibration_iterations=10 # independent calibrations "runs" for stats/consistency purposes
    flasher_rate_Hz=10
    no_of_flashes_in_calibration_step=10
    rise_time_s=500 # time for NSB to be become maximal (and conversely fall back to the minimum NSB)
    min_NSB_MHz=50
    max_NSB_MHz=1000
    take_frequent_pedestal = True
    no_of_calibration_steps = 10

    PlotLabel = "vary_time_between_flashes_with_sine_wave_nsb"
    output_file="/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/" + PlotLabel + '.log'

    # investigate in relation to multiples of rise time
    for interval_as_fraction_of_rise_time in [0.1,0.25,0.5,0.75,1,1.1]:

    #for interval_as_fraction_of_rise_time in [0.1,0.5,0.75,1]:
        interval_between_calibration_s = interval_as_fraction_of_rise_time*rise_time_s
        print("*****************")
        print("interval_as_fraction_of_rise_time " + str(interval_as_fraction_of_rise_time))
        PrintParams(output_file,no_of_calibration_iterations, nsb_MHz_at_time_zero, flasher_illumination_time_zero,
                    nsb_rate_of_change_MHz_s, flasher_rate_Hz, no_of_flashes_in_calibration_step,
                    interval_between_calibration_s)

        mean_MPRCs,std_dev_MPRCs,RMSEs,mean_NSBs,mean_TPCs= run_calibration_sine_wave_NSB(take_frequent_pedestal, output_file, no_of_calibration_iterations, \
                                                            flasher_illumination_time_zero, flasher_rate_Hz, no_of_calibration_steps, \
                                                            no_of_flashes_in_calibration_step, interval_between_calibration_s, \
                                                            min_NSB_MHz, max_NSB_MHz, rise_time_s)

        plotnormal(ax1, mean_TPCs,mean_MPRCs, "Interval as fraction of rise time " +str(interval_as_fraction_of_rise_time) )
        plotnormal(ax2, mean_TPCs, RMSEs,"Interval as fraction of rise time " +str(interval_as_fraction_of_rise_time))
        plotnormal(ax3, mean_TPCs, std_dev_MPRCs, "Interval as fraction of rise time " +str(interval_as_fraction_of_rise_time))

    # show_plt(ax1)
    fname = "/home/sheridan/Documents/SSTCAM/TO PRESENT/JON_L_LASER_INVESTIGATION/" + PlotLabel + '.png'
    #plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait',  format=None,
    #            transparent=False, bbox_inches=None, pad_inches=0.1)

    show_plt(ax1)
    show_plt(ax2)
    show_plt(ax3)

    return
#PltVaryTimeBetweenFlashes()