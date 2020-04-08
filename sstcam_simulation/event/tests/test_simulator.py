from sstcam_simulation.camera import Camera, PixelMapping
from sstcam_simulation.event import PhotoelectronSource


def test_event_simulator():
    camera = Camera()
    PhotoelectronSource(camera)


def test_get_nsb():
    pixel_mapping = PixelMapping(n_pixels=2)
    camera = Camera(pixel=pixel_mapping)
    simulator = PhotoelectronSource(camera, seed=1)
    nsb = simulator.get_nsb(rate=1000)
    assert (nsb.pixel.size == nsb.time.size) & (nsb.pixel.size == nsb.charge.size)
    assert (nsb.pixel == 0).sum() == 1001
    assert (nsb.pixel == 1).sum() == 982
    assert (nsb.time > 0).all()
    assert (nsb.charge > 0).all()


def test_seed():
    camera = Camera()

    simulator_1 = PhotoelectronSource(camera)
    simulator_2 = PhotoelectronSource(camera)
    simulator_3 = PhotoelectronSource(camera, seed=simulator_2.seed)
    simulator_4 = PhotoelectronSource(camera, seed=1)
    simulator_5 = PhotoelectronSource(camera, seed=1)
    simulator_6 = PhotoelectronSource(camera, seed=2)

    # get_nsb
    nsb_1 = simulator_1.get_nsb(rate=1000)
    nsb_2 = simulator_2.get_nsb(rate=1000)
    nsb_3 = simulator_3.get_nsb(rate=1000)
    nsb_4 = simulator_4.get_nsb(rate=1000)
    nsb_5 = simulator_5.get_nsb(rate=1000)
    nsb_6 = simulator_6.get_nsb(rate=1000)
    assert nsb_1 != nsb_2
    assert nsb_2 != nsb_3  # Generator has been progressed
    assert nsb_3 != nsb_4
    assert nsb_4 == nsb_5
    assert nsb_5 != nsb_6
