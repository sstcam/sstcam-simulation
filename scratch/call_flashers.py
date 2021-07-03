import sys
sys.path.append(r'/home/sheridan/sstcam-simulation')


from sstcam_simulation.event.source import PhotoelectronSource
from sstcam_simulation.camera import Camera
from sstcam_simulation.camera import SSTCameraMapping
from sstcam_simulation.event.acquisition import EventAcquisition
from sstcam_simulation.camera.noise import GaussianNoise
from sstcam_simulation.plotting.image import CameraImage
from matplotlib import pyplot as plt
import numpy as np

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


#single_flasher_call_no_err()
#single_flasher_call_with_err(time=2, illumination=500, flasher_pulse_width=4.5, illumination_err=0, pulse_width_err=0)
#single_flasher_call_with_err(time=2, illumination=500, flasher_pulse_width=4.5, illumination_err=0.2, pulse_width_err=0.035)
#single_flasher_call_with_err(time=2, illumination=500, flasher_pulse_width=4.5, illumination_err=0.5, pulse_width_err=0.035)
#single_flasher_call_with_err(time=2, illumination=500, flasher_pulse_width=4.5, illumination_err=0.5, pulse_width_err=0.2)
# get_flasher_illumination(self, time, illumination, flasher_pulse_width, illumination_err=0, pulse_width_err=0):

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

#plot_multiple_flasher(illumination_err=0, pulse_width_err=0)

#plot_multiple_flasher(illumination_err=0.2, pulse_width_err=0.035)
#plot_multiple_flasher(illumination_err=0, pulse_width_err=0)


# test get uniform illumination - was altered for flashers
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


#FlasherChargeResolutionVaryTimePlotWaveFormWithNSB()


# 12/6/21
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
#print("Flash_100_200_pe_nsb_no_correction")
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
print ("26 June Refactor code")
# 26 June 2021 - Refactor code - Now make the above more configurable so we can seperate iteration conditions better
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

    true_charge = flasher.get_photoelectrons_per_pixel(n_pixels)

    # Add in NSB
    nsb = source.get_nsb(rate=nsb_MHz)
    pe = flasher + nsb

    readout = acquisition.get_continuous_readout(pe)
    waveform = acquisition.get_sampled_waveform(readout)

    # Charge Extraction
    measured_charge = waveform.sum(1)
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
target_fractional_err=0.015 # target fractional error , defined as (measured-true)/true
no_of_flashes_step=1 # reach goal quicker and accept some imprecision in number of flashes - this actually isnt't so usefull - increasing step size doesn't seem to help much

def PlotFractionalErr(no_of_nsb_obs,no_of_flashes, flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err):

    for flasher_illumination in flasher_illuminations:
        print("Flasher Illumination " + str(flasher_illumination) + " p.e. ")
        for nsb_MHz in nsb_rates:
            print("NSB  " + str(nsb_MHz) + " MHz ")
            nsb_pedestal_charge=get_nsb_pedestal(nsb_MHz, no_of_nsb_obs)
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

flasher_illuminations=[100]
print ("PlotFractionalErr 100 pe")
PlotFractionalErr(no_of_nsb_obs,no_of_flashes, flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err)

flasher_illuminations=[200]
print ("PlotFractionalErr pe")
PlotFractionalErr(no_of_nsb_obs,no_of_flashes, flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err)


# no_of_runs - how many callibrations to attempt - we should achieve target_fractional_err consistently in this consecutive number of calibration runs
def GetNumberOfFlashesToReachTargetFractionalError(no_of_flashes_step,no_of_consistent_runs,target_fractional_err,no_of_nsb_obs,flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err):
    print ("Begin GetNumberOfFlashesToReachTargetFractionalError - no of NSB obs " + str(no_of_nsb_obs))
    for flasher_illumination in flasher_illuminations:
        #print("Flasher Illumination " + str(flasher_illumination) + " p.e. ")
        for nsb_MHz in nsb_rates:
            fractional_errors_n_runs = []
            achieved_target_fractional_err=False
            no_of_flashes=0
            while (achieved_target_fractional_err==False):
                no_of_flashes+=no_of_flashes_step
                #print ("no_of_flashes " + str(no_of_flashes))
                debug_fract_errs=[]

                for x in range (no_of_consistent_runs): # we need to meet the target err consistently in this number of runs

                    nsb_pedestal_charge=get_nsb_pedestal(nsb_MHz, no_of_nsb_obs)
                    fractional_err_charge_moving_avgs=[]
                    event_nos=[]
                    event_nos,fractional_err_charge_moving_avgs=get_fractional_err_flashers_moving_avg(no_of_flashes, nsb_MHz, nsb_pedestal_charge, flasher_illumination,
                                                           pulse_width, illumination_err, pulse_width_err)
                    debug_fract_errs.append(fractional_err_charge_moving_avgs[-1])

                    if (fractional_err_charge_moving_avgs[-1] > target_fractional_err): # get the last fractional error after no_of_flashes
                        # this is the failure case so don't bother to iterate further
                        #print ("Fractional error, " + str(fractional_err_charge_moving_avgs[-1]) + " ,Breach at run " + str(x)  + " with " + str(no_of_flashes) + " flashes ")
                        break
                if (abs(fractional_err_charge_moving_avgs[-1]) > target_fractional_err):
                    continue
                else:
                    #print ("run " + str(x))
                    achieved_target_fractional_err = True

            print("Illumination," + str(flasher_illumination) + ",NSB," + str(nsb_MHz) + ",Target Fractional error," + str(target_fractional_err) + " achieved for " + str(no_of_consistent_runs) + " runs in " + str(no_of_flashes) + " flashes")
    return
print ("GetNumberOfFlashesToReachTargetFractionalError")
GetNumberOfFlashesToReachTargetFractionalError(no_of_flashes_step,no_of_consistent_runs,target_fractional_err,no_of_nsb_obs,flasher_illuminations, nsb_rates,pulse_width, illumination_err, pulse_width_err)
