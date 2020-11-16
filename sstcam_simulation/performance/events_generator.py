from sstcam_simulation import PhotoelectronSource, EventAcquisition, PhotoelectronReader
from sstcam_simulation.data import get_data
from abc import ABCMeta, abstractmethod
import tables
from os.path import exists


class EventsGenerator(metaclass=ABCMeta):
    def __init__(self, camera, extractor, pedestal, nsb_rate):
        self.camera = camera
        self.extractor = extractor
        self.pedestal = pedestal
        self.nsb_rate = nsb_rate

        pulse_area = self.camera.photoelectron_pulse.area
        spectrum_average = self.camera.photoelectron_spectrum.average
        pe_conversion = pulse_area * spectrum_average
        self.pe_conversion = pe_conversion
        self.n_pixels = self.camera.mapping.n_pixels
        self.n_superpixels = self.camera.mapping.n_superpixels

        self.signal_time = 40  # ns
        self.signal_index = self.camera.get_waveform_sample_from_time(self.signal_time)
        self.source = PhotoelectronSource(camera=self.camera)
        self.acquisition = EventAcquisition(camera=self.camera)

    @property
    @abstractmethod
    def event_table_layout(self):
        """
        Define the event table layout, matching the contents of the returned
        dict from __iter__

        Returns
        -------
        Class
            Subclass of tables.IsDescription
        """

    @abstractmethod
    def generate_event(self):
        """
        Generate the event and extract the useful event information

        Returns
        -------
        dict
            Dict containing the return values, matching the specification
            in event_table_layout
        """


class LabIlluminationGenerator(EventsGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.illumination = 50

    def set_illumination(self, illumination):
        self.illumination = illumination

    @property
    def event_table_layout(self):
        class EventTable(tables.IsDescription):
            n_triggers = tables.Int64Col(shape=self.n_superpixels)
            true_charge = tables.Int64Col(shape=self.n_pixels)
            measured_charge = tables.Float64Col(shape=self.n_pixels)

            # Event metadata
            illumination = tables.Float64Col()
        return EventTable

    def generate_event(self):
        nsb_pe = self.source.get_nsb(self.nsb_rate)
        signal_pe = self.source.get_uniform_illumination(
            self.signal_time, self.illumination, laser_pulse_width=3
        )
        true_charge = signal_pe.get_photoelectrons_per_pixel(self.n_pixels)
        readout = self.acquisition.get_continuous_readout(nsb_pe + signal_pe)

        trigger = self.acquisition.trigger
        digital_trigger = trigger.get_superpixel_digital_trigger_line(readout)
        n_triggers = trigger.get_n_superpixel_triggers(digital_trigger)

        waveform = self.acquisition.get_sampled_waveform(readout)
        measured_charge = self.extractor.extract(waveform, self.signal_index)
        calibrated_charge = (measured_charge - self.pedestal) / self.pe_conversion

        return dict(
            n_triggers=n_triggers,
            true_charge=true_charge,
            measured_charge=calibrated_charge,
            illumination=self.illumination,
        )


class CherenkovShowerGenerator(EventsGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not exists(self.path):
            msg = '''
            Cherenkov files have not been downloaded to sstcam_simulation/data/cherenkov. 
            The files can be downloaded from Nextcloud 
            https://pcloud.mpi-hd.mpg.de/index.php/f/142621
            '''
            raise ValueError(msg)
        self.reader = PhotoelectronReader(self.path)

        if self.n_pixels != 2048:
            print("Warning: Full camera pixels not simulated for cherenkov events")

    @property
    @abstractmethod
    def path(self):
        pass

    @property
    def event_table_layout(self):
        class EventTable(tables.IsDescription):
            n_triggers = tables.Int64Col(shape=1)
            true_charge = tables.Int64Col(shape=self.camera.mapping.n_pixels)
            measured_charge = tables.Float64Col(shape=self.camera.mapping.n_pixels)

            # Event metadata
            event_index = tables.UInt64Col()
            event_id = tables.UInt64Col()
            telescope_id = tables.UInt8Col()
            n_photoelectrons = tables.UInt64Col()
            energy = tables.Float64Col()
            alt = tables.Float64Col()
            az = tables.Float64Col()
            core_x = tables.Float64Col()
            core_y = tables.Float64Col()
            h_first_int = tables.Float64Col()
            x_max = tables.Float64Col()
            shower_primary_id = tables.UInt8Col()
        return EventTable

    def generate_event(self):
        cherenkov_pe = self.reader.random_event()
        nsb_pe = self.source.get_nsb(self.nsb_rate)
        signal_pe = self.source.resample_photoelectron_charge(cherenkov_pe)
        true_charge = signal_pe.get_photoelectrons_per_pixel(self.n_pixels)
        readout = self.acquisition.get_continuous_readout(nsb_pe + signal_pe)

        n_triggers = self.acquisition.get_trigger(readout).size

        waveform = self.acquisition.get_sampled_waveform(readout)
        measured_charge = self.extractor.extract(waveform, self.signal_index)
        calibrated_charge = (measured_charge - self.pedestal) / self.pe_conversion

        return dict(
            n_triggers=n_triggers,
            true_charge=true_charge,
            measured_charge=calibrated_charge,
            **signal_pe.metadata,
        )


class GammaShowerGenerator(CherenkovShowerGenerator):
    @property
    def path(self):
        return get_data("cherenkov/gamma.h5")


class ProtonShowerGenerator(CherenkovShowerGenerator):
    @property
    def path(self):
        return get_data("cherenkov/proton.h5")
