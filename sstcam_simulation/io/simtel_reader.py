from sstcam_simulation.event.photoelectrons import Photoelectrons
from sstcam_simulation.camera.mapping import SSTCameraMapping
from eventio import SimTelFile
import numpy as np
from CHECLabPy.utils.mapping import get_row_column


__all__ = ["SimtelReader", "get_pixel_remap"]


def get_pixel_remap(pixel_x, pixel_y, coordinates):
    """

    Parameters
    ----------
    pixel_x : ndarray
        X coordinates of pixels to be remapped
    pixel_y : ndarray
        Y coordinates of pixels to be remapped
    coordinates : CameraCoordinates
        sstcam-simulation pixel coordinates object
        provides destination pixel mapping

    Returns
    -------

    """
    row, col = get_row_column(pixel_x, pixel_y)
    n_rows = row.max() + 1
    n_columns = col.max() + 1
    n_pixels = pixel_x.size
    camera_2d = np.zeros((n_rows, n_columns), dtype=np.int)
    camera_2d[coordinates.row, coordinates.column] = np.arange(n_pixels, dtype=np.int)
    return camera_2d[row, col]


class SimtelReader:
    def __init__(
            self,
            path: str,
            disable_remapping: bool = False,
            only_triggered_events: bool = False,
            n_events: int = None,
    ):
        """
        Read Photoelectron arrays directly from simtelarray files

        Parameters
        ----------
        path : str
            Path to the simtel file
        disable_remapping : bool
            Disables the remapping of the pixels to the sstcam-simulation
            pixel mapping
        only_triggered_events : bool
            Only read events which caused a telescope trigger
        n_events : int
            Number of telescope events to process
        """
        self._file = SimTelFile(path)
        self._disable_remapping = disable_remapping
        self._only_triggered_events = only_triggered_events
        self._n_events = n_events
        self.n_pixels = 2048

        self._camera_settings = {}
        self._pixel_remap = {}
        mapping = SSTCameraMapping()
        for telid, tel in self._file.telescope_descriptions.items():
            camera_settings = tel['camera_settings']
            self._camera_settings[telid] = camera_settings
            self._pixel_remap[telid] = get_pixel_remap(
                camera_settings['pixel_x'],
                camera_settings['pixel_y'],
                mapping.pixel
            )

    @property
    def camera_settings(self):
        return self._camera_settings

    @property
    def pixel_remap(self):
        return self._pixel_remap

    def __iter__(self):
        n_events = 0
        if self._only_triggered_events:
            it = self._file.iter_array_events()
        else:
            it = self._file.iter_mc_events()
        for iev, event in enumerate(it):
            if 'photoelectrons' not in event:
                continue

            photoelectrons = event['photoelectrons']
            mc_shower = event['mc_shower']
            mc_event = event['mc_event']

            if self._only_triggered_events:
                tel_ids = event['telescope_events'].keys()
            else:
                tel_ids = np.array(list(photoelectrons.keys())) + 1
            for tel_id in tel_ids:
                # Retrieve only SST Camera
                if self._camera_settings[tel_id]['n_pixels'] != self.n_pixels:
                    continue

                n_events += 1
                if self._n_events and n_events > self._n_events:
                    return

                values = photoelectrons[tel_id-1]
                metadata = dict(
                    event_index=iev,
                    event_id=event['event_id'],
                    telescope_id=tel_id,
                    n_photoelectrons=values['time'].size,
                    energy=mc_shower["energy"],
                    alt=mc_shower["altitude"],
                    az=mc_shower["azimuth"],
                    core_x=mc_event["xcore"],
                    core_y=mc_event["ycore"],
                    h_first_int=mc_shower["h_first_int"],
                    x_max=mc_shower["xmax"],
                    shower_primary_id=mc_shower["primary_id"]
                )

                pixel = values['pixel_id']
                time = values['time']
                charge = np.ones(pixel.size)

                # Shift photoelectron times to sensible reference time
                start_time = 30  # ns
                time = start_time + time - time.min()

                # Convert pixel mapping
                if not self._disable_remapping:
                    pixel = self._pixel_remap[tel_id][pixel]

                yield Photoelectrons(
                    pixel=pixel, time=time, charge=charge, metadata=metadata
                )
