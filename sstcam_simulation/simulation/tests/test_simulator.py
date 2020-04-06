from sstcam_simulation.camera import Camera
from sstcam_simulation.simulation import EventSimulator


def test_event_simulator():
    camera = Camera()
    EventSimulator(camera)


def test_get_nsb():
    camera = Camera(n_pixels=2)
    simulator = EventSimulator(camera, seed=1)
    nsb = simulator.get_nsb(rate=1000)
    assert (nsb.pixel.size == nsb.time.size) & (nsb.pixel.size == nsb.charge.size)
    assert (nsb.pixel == 0).sum() == 128
    assert (nsb.pixel == 1).sum() == 122
    assert (nsb.time > 0).all()
    assert (nsb.charge > 0).all()


def test_seed():
    camera = Camera()

    simulator_1 = EventSimulator(camera)
    simulator_2 = EventSimulator(camera)
    simulator_3 = EventSimulator(camera, seed=simulator_2.seed)
    simulator_4 = EventSimulator(camera, seed=1)
    simulator_5 = EventSimulator(camera, seed=1)
    simulator_6 = EventSimulator(camera, seed=2)

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
