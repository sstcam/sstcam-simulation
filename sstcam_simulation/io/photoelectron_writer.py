import tables
from typing import Type
from sstcam_simulation.event.photoelectrons import Photoelectrons

FILTERS = tables.Filters(
    complevel=5,  # compression medium, tradeoff between speed and compression
    complib="blosc:zstd",  # use modern zstd algorithm
    fletcher32=True,  # add checksums to data chunks
)


class PhotoelectronWriter:
    def __init__(self, path: str, event_table_desc: Type[tables.IsDescription]):
        """
        Writer for a collection of Photoelectrons to HDF5 file using pytables

        Stores the photoelectron info as variable length arrays

        Parameters
        ----------
        path : str
            Path to store the file (overwrites if already exists)
        event_table_desc : Type[tables.IsDescription]
            Uninstanced `tables.IsDescription` class describing the columns of
            the event metadata table
        """
        self._file = tables.File(path, mode='w', filters=FILTERS)
        group = self._file.create_group(self._file.root, "data", "Event Data")
        self._event_metadata_table = self._file.create_table(
            group, "event_metadata", event_table_desc, "Event Metadata"
        )
        self._pixel_column = self._file.create_vlarray(
            group,
            "photoelectron_arrival_pixel",
            tables.UInt16Atom(shape=()),
            "Pixel hit by the photoelectron"
        )
        self._time_column = self._file.create_vlarray(
            group,
            "photoelectron_arrival_time",
            tables.Float64Atom(shape=()),
            "Arrival time of the photoelectrons"
        )
        self._charge_column = self._file.create_vlarray(
            group,
            "photoelectron_measured_charge",
            tables.Float64Atom(shape=()),
            "Charge reported by photosensor for each photoelectron"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._file.close()

    def flush(self):
        self._file.flush()

    def append(self, pe: Photoelectrons):
        """
        Append a Photoelectrons object to the file as a new event

        Parameters
        ----------
        pe : Photoelectrons
            The photoelectrons to append
        """
        row = self._event_metadata_table.row
        for key, value in pe.metadata.items():
            row[key] = value
        row.append()

        self._pixel_column.append(pe.pixel)
        self._time_column.append(pe.time)
        self._charge_column.append(pe.charge)
