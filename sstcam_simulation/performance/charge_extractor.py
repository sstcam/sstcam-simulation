from scipy import interpolate
from scipy.ndimage import correlate1d
from ctapipe.instrument import CameraGeometry
from ctapipe.image.extractor import neighbor_average_waveform, extract_around_peak
from astropy import units as u
import numpy as np


class ChargeExtractor:
    def __init__(self, ref_x, ref_y, mapping):
        # Prepare for cc
        f = interpolate.interp1d(ref_x, ref_y, kind=3)
        cc_ref_x = np.arange(0, ref_x[-1], 1)
        cc_ref_y = f(cc_ref_x)
        y_1pe = cc_ref_y / np.trapz(cc_ref_y)
        self.origin = cc_ref_y.argmax() - cc_ref_y.size // 2
        scale = correlate1d(y_1pe, cc_ref_y, mode='constant', origin=self.origin).max()
        self.cc_ref_y = cc_ref_y / scale

        self.neighbours = CameraGeometry(
            "sstcam",
            mapping.pixel.i,
            u.Quantity(mapping.pixel.x, 'm'),
            u.Quantity(mapping.pixel.y, 'm'),
            u.Quantity(mapping.pixel.size, 'm')**2,
            'square'
        ).neighbor_matrix_sparse

    @classmethod
    def from_camera(cls, camera):
        ref_x = camera.photoelectron_pulse.time
        ref_y = camera.photoelectron_pulse.amplitude
        mapping = camera.mapping
        return cls(ref_x, ref_y, mapping)

    def extract(self, waveforms, peak_index):
        cc = correlate1d(waveforms, self.cc_ref_y, mode='constant', origin=self.origin)
        charge, _, = extract_around_peak(cc, peak_index, 1, 0, 1)
        return charge

    def obtain_peak_index_from_neighbours(self, waveforms):
        average_wfs = neighbor_average_waveform(
            waveforms,
            neighbors_indices=self.neighbours.indices,
            neighbors_indptr=self.neighbours.indptr,
            lwt=0,
        )
        return average_wfs.argmax(axis=-1)
