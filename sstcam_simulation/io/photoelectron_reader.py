from sstcam_simulation import Photoelectrons
import tables


class PhotoelectronReader:
    def __init__(self, path: str):
        """
        Reader for photoelectrons stored to a HDF5 file

        Parameters
        ----------
        path : str
            Path to the HDF5 file
        """
        self._file = tables.File(path, mode='r')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._file.close()

    def __iter__(self):
        data = self._file.root.data
        table_iter = data.event_metadata.iterrows()
        pe_pixel_iter = data.photoelectron_arrival_pixel.iterrows()
        pe_time_iter = data.photoelectron_arrival_time.iterrows()
        pe_charge_iter = data.photoelectron_measured_charge.iterrows()
        event_iter = zip(table_iter, pe_pixel_iter, pe_time_iter, pe_charge_iter)

        for row, pixel, time, charge in event_iter:
            metadata = {key: row[key] for key in row.table.colnames}
            yield Photoelectrons(
                pixel=pixel, time=time, charge=charge, metadata=metadata
            )
