from sstcam_simulation.io.simtel_reader import SimtelReader, get_pixel_remap
from sstcam_simulation.data import get_data
from sstcam_simulation.camera.mapping import CameraCoordinates, SSTCameraMapping
from ctapipe.io import SimTelEventSource
import numpy as np


def test_get_pixel_remap():
    pixel_x = np.array([-0.2, -0.2, 0, 0, 0.2, 0.2, -0.2, 0, 0.2])
    pixel_y = np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0, 0, 0])

    coords = CameraCoordinates(
        i=None,
        x=np.array([0, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0, 0]),
        y=np.array([-0.1, -0.1, -0.1, 0.1, 0.1, 0, 0, 0.1, 0]),
        row=np.array([0, 0, 0, 2, 2, 1, 1, 2, 1]),
        column=np.array([1, 2, 0, 2, 0, 2, 0, 1, 1]),
        neighbours=None,
        size=None
    )
    pixel_sets = [(pixel_x[i]/2, pixel_y[i]/2) for i in range(pixel_x.size)]
    coord_sets = [(coords.x[i], coords.y[i]) for i in range(pixel_x.size)]
    remap = get_pixel_remap(pixel_x, pixel_y, coords)
    expected = np.array([coord_sets.index(p) for p in pixel_sets])
    assert np.array_equal(remap, expected)


def test_simtelreader():
    path = get_data("testing/simtel_test.simtel.gz")
    reader = SimtelReader(path, disable_remapping=False, only_triggered_events=False)
    photoelectrons = []
    for pe in reader:
        photoelectrons.append(pe)

    assert len(photoelectrons) == 144
    for pe in photoelectrons:
        assert len(pe.pixel) == len(pe.time) == len(pe.charge)
    assert len(photoelectrons[0].pixel) == 24
    assert photoelectrons[0].time.min() == 30
    assert (photoelectrons[0].charge == 1).all()

    reader = SimtelReader(path, disable_remapping=False, only_triggered_events=True)
    photoelectrons = []
    for pe in reader:
        photoelectrons.append(pe)

    assert len(photoelectrons) == 15
    assert len(photoelectrons[0].pixel) == 86


def test_pixel_remapping():
    pass  #TODO


def test_comparison_to_ctapipe_true_image():
    path = get_data("testing/simtel_test.simtel.gz")

    source = SimTelEventSource(input_url=path)
    ctapipe_images = {}
    for event in source:
        for telid, tel in event.mc.tel.items():
            key = (event.index['event_id'], telid)
            ctapipe_images[key] = tel.true_image.astype(np.int)

    reader = SimtelReader(path, disable_remapping=True, only_triggered_events=True)
    photoelectron_images = {}
    for pe in reader:
        key = (pe.metadata['event_id'], pe.metadata['telescope_id'])
        photoelectron_images[key] = pe.get_photoelectrons_per_pixel(reader.n_pixels)

    assert ctapipe_images.keys() == photoelectron_images.keys()
    for key in ctapipe_images.keys():
        assert np.array_equal(ctapipe_images[key], photoelectron_images[key])

    # Check the triggered-events are still correct when reading the non-triggered
    reader = SimtelReader(path, disable_remapping=True, only_triggered_events=False)
    photoelectron_images = {}
    for pe in reader:
        key = (pe.metadata['event_id'], pe.metadata['telescope_id'])
        photoelectron_images[key] = pe.get_photoelectrons_per_pixel(reader.n_pixels)
    for key in ctapipe_images.keys():
        assert np.array_equal(ctapipe_images[key], photoelectron_images[key])


def test_simtel_reader_pixel_remap():
    path = get_data("testing/simtel_test.simtel.gz")
    reader = SimtelReader(path, disable_remapping=True, only_triggered_events=True)
    events_original = list(reader)
    pixel_x_original = reader.camera_settings[1]['pixel_x']
    pixel_y_original = reader.camera_settings[1]['pixel_y']

    reader = SimtelReader(path, disable_remapping=False, only_triggered_events=True)
    events_remapped = list(reader)
    mapping = SSTCameraMapping()
    pixel_x_remapped = mapping.pixel.x
    pixel_y_remapped = mapping.pixel.y

    for pe_original, pe_remapped in zip(events_original, events_remapped):
        image_original = pe_original.get_photoelectrons_per_pixel(reader.n_pixels)
        cog_x_original = np.average(pixel_x_original, weights=image_original)
        cog_y_original = np.average(pixel_y_original, weights=image_original)

        image_remapped = pe_remapped.get_photoelectrons_per_pixel(reader.n_pixels)
        cog_x_remapped = np.average(pixel_x_remapped, weights=image_remapped)
        cog_y_remapped = np.average(pixel_y_remapped, weights=image_remapped)

        np.testing.assert_allclose(cog_x_original, cog_x_remapped, rtol=1e-4)
        np.testing.assert_allclose(cog_y_original, cog_y_remapped, rtol=1e-4)


def test_n_events():
    path = get_data("testing/simtel_test.simtel.gz")
    reader = SimtelReader(path, n_events=3)
    assert(len(list(reader)) == 3)
