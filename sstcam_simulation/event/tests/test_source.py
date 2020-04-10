from sstcam_simulation.camera import Camera, PixelMapping
from sstcam_simulation.event import PhotoelectronSource
import numpy as np


def test_photoelectron_source():
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


def test_get_uniform_illumination():
    pixel_mapping = PixelMapping(n_pixels=2)
    camera = Camera(pixel=pixel_mapping)
    simulator = PhotoelectronSource(camera, seed=1)
    pe = simulator.get_uniform_illumination(time=40, illumination=100, pulse_width=2)
    assert (pe.pixel.size == pe.time.size) & (pe.pixel.size == pe.charge.size)
    assert (pe.pixel == 0).sum() == 100
    assert (pe.pixel == 1).sum() == 94
    assert (pe.time > 0).all()
    assert (pe.charge > 0).all()
    np.testing.assert_allclose(pe.time.mean(), 40, rtol=1e-1)
    np.testing.assert_allclose(pe.time.std(), 2, rtol=1e-1)


def test_seed():
    camera = Camera()

    simulator_1 = PhotoelectronSource(camera)
    simulator_2 = PhotoelectronSource(camera)
    simulator_3 = PhotoelectronSource(camera, seed=simulator_2.seed)
    simulator_4 = PhotoelectronSource(camera, seed=1)
    simulator_5 = PhotoelectronSource(camera, seed=1)
    simulator_6 = PhotoelectronSource(camera, seed=2)

    # get_nsb
    sim_1 = simulator_1.get_nsb(rate=1000)
    sim_2 = simulator_2.get_nsb(rate=1000)
    sim_3 = simulator_3.get_nsb(rate=1000)
    sim_4 = simulator_4.get_nsb(rate=1000)
    sim_5 = simulator_5.get_nsb(rate=1000)
    sim_6 = simulator_6.get_nsb(rate=1000)
    assert sim_1 != sim_2
    assert sim_2 != sim_3  # Generator has been progressed
    assert sim_3 != sim_4
    assert sim_4 == sim_5
    assert sim_5 != sim_6

    # get_uniform_illumination
    sim_1 = simulator_1.get_uniform_illumination(time=40, illumination=50, pulse_width=2)
    sim_2 = simulator_2.get_uniform_illumination(time=40, illumination=50, pulse_width=2)
    sim_3 = simulator_3.get_uniform_illumination(time=40, illumination=50, pulse_width=2)
    sim_4 = simulator_4.get_uniform_illumination(time=40, illumination=50, pulse_width=2)
    sim_5 = simulator_5.get_uniform_illumination(time=40, illumination=50, pulse_width=2)
    sim_6 = simulator_6.get_uniform_illumination(time=40, illumination=50, pulse_width=2)
    assert sim_1 != sim_2
    assert sim_2 != sim_3  # Generator has been progressed
    assert sim_3 != sim_4
    assert sim_4 == sim_5
    assert sim_5 != sim_6
