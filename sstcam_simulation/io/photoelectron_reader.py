from sstcam_simulation import Photoelectrons
import numpy as np
import tables


class PhotoelectronReader:
    def __init__(self, path: str, seed=None):
        """
        Reader for photoelectrons stored to a HDF5 file

        Parameters
        ----------
        path : str
            Path to the HDF5 file
        seed : int or tuple
            Seed for the numpy random number generator.
            Ensures the reproducibility of an event if you know its seed
        """
        self._file = tables.File(path, mode='r')
        self._rng = np.random.RandomState(seed=seed)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        return self._file.root.data.event_metadata.attrs.NROWS

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

    def __getitem__(self, item):
        data = self._file.root.data
        pixel = data.photoelectron_arrival_pixel[item]
        time = data.photoelectron_arrival_time[item]
        charge = data.photoelectron_measured_charge[item]
        metadata = dict(zip(data.event_metadata.colnames, data.event_metadata[item]))
        return Photoelectrons(pixel=pixel, time=time, charge=charge, metadata=metadata)

    def random_event(self, poisson_fluctuate=False):
        """
        Obtain a random Cherenkov event from the file

        Parameters
        ----------
        poisson_fluctuate : bool
            Also assume the photoelectrons are an "average" representation for
            this shower, and fluctuate them with a Poisson distribution.
            Can be used to get a different image for the same event, but is not
            technically correct.

        Returns
        -------
        pe : Photoelectrons
            Initial photoelectrons generated in a camera upon observing a Cherenkov shower
        """
        pe = self[self._rng.randint(0, self.__len__(), 1)[0]]
        if poisson_fluctuate:
            repeats = self._rng.poisson(1, len(pe))
            return Photoelectrons(
                pixel=np.repeat(pe.pixel, repeats),
                time=np.repeat(pe.time, repeats),
                charge=np.ones(repeats.sum()),
                metadata=pe.metadata
            )
        else:
            return pe
