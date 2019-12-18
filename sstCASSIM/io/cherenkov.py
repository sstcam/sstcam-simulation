"""
IO classes for the sstCASSIM event-list HDF5 file

Structure - Each entry is a photon:
event pixel time
OR (TODO: decide)
Photons are bunched:
event pixel time n_photons

TODO:
    * Serial or random access?
"""


class CherenkovWriter:
    def __init__(self):
        """
        Writer for the sstCASSIM event-list HDF5 file.
        """
        pass

    def write_event(self, event):
        pass


class CherenkovReader:
    def __init__(self):
        """
        Reader for the sstCASSIM event-list HDF5 file.
        """
        pass

    def read_next_event(self):
        pass
