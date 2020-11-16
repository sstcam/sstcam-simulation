from sstcam_simulation import PhotoelectronSource, EventAcquisition
import numpy as np
from tqdm import trange


def obtain_pedestal(camera, extractor, nsb_rate=40):
    print("Obtaining pedestal")

    # Update the AC coupling for this nsb rate
    camera.coupling.update_nsb_rate(nsb_rate)

    source = PhotoelectronSource(camera=camera)
    acquisition = EventAcquisition(camera=camera)

    n_empty = 100
    pedestal_array = np.zeros((n_empty, camera.mapping.n_pixels))
    for i in trange(n_empty, desc="Measuring pedestal"):
        nsb_pe = source.get_nsb(nsb_rate)
        readout = acquisition.get_continuous_readout(nsb_pe)
        waveform = acquisition.get_sampled_waveform(readout)
        charge = extractor.extract(waveform, camera.n_waveform_samples//2)
        pedestal_array[i] = charge
    return np.median(pedestal_array)
