from sstcam_simulation.camera import Camera, SSTCameraMapping
from sstcam_simulation.event import PhotoelectronSource, Photoelectrons, EventAcquisition
import numpy as np


def test_photoelectron_source():
    camera = Camera()
    PhotoelectronSource(camera)


def test_get_nsb():
    mapping = SSTCameraMapping(n_pixels=2)
    camera = Camera(mapping=mapping)
    simulator = PhotoelectronSource(camera, seed=1)
    nsb = simulator.get_nsb(rate=1000)
    assert (nsb.pixel.size == nsb.time.size) & (nsb.pixel.size == nsb.charge.size)
    assert (nsb.pixel == 0).sum() == 1001
    assert (nsb.pixel == 1).sum() == 982
    assert (nsb.time > 0).all()
    assert (nsb.charge > 0).all()


def test_get_uniform_illumination():
    mapping = SSTCameraMapping(n_pixels=2)
    camera = Camera(mapping=mapping)
    simulator = PhotoelectronSource(camera, seed=1)
    pe = simulator.get_uniform_illumination(
        time=40, illumination=100, laser_pulse_width=2
    )
    assert (pe.pixel.size == pe.time.size) & (pe.pixel.size == pe.charge.size)
    assert (pe.pixel == 0).sum() == 100
    assert (pe.pixel == 1).sum() == 94
    assert (pe.time > 0).all()
    assert (pe.charge > 0).all()
    np.testing.assert_allclose(pe.time.mean(), 40, rtol=1e-1)
    np.testing.assert_allclose(pe.time.std(), 2, rtol=1e-1)


def test_get_cherenkov_shower():
    shower_kwargs = dict(
        centroid_x=0.1,
        centroid_y=0,
        length=0.01,
        width=0.01,
        psi=0,
        time_gradient=0,
        time_intercept=20,
        intensity=1000,
        cherenkov_pulse_width=5,
    )

    camera = Camera()
    simulator = PhotoelectronSource(camera, seed=1)
    pe = simulator.get_cherenkov_shower(**shower_kwargs)
    assert (pe.pixel.size == pe.time.size) & (pe.pixel.size == pe.charge.size)
    assert pe.pixel.size == 991
    assert (pe.charge > 0).all()
    np.testing.assert_allclose(pe.charge.sum(), 991, rtol=1e-1)
    np.testing.assert_allclose(pe.time.mean(), 20, rtol=1e-1)
    np.testing.assert_allclose(pe.time.std(), 5, rtol=1e-1)


def test_get_random_cherenkov_shower():
    camera = Camera()
    simulator = PhotoelectronSource(camera, seed=1)
    pe = simulator.get_random_cherenkov_shower(cherenkov_pulse_width=5)
    assert (pe.pixel.size == pe.time.size) & (pe.pixel.size == pe.charge.size)
    assert pe.pixel.size == 40891
    assert (pe.charge > 0).all()
    np.testing.assert_allclose(pe.charge.sum(), 41032.3, rtol=1e-1)
    np.testing.assert_allclose(pe.time.mean(), 827.86, rtol=1e-1)
    np.testing.assert_allclose(pe.time.std(), 5, rtol=1e-1)


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
    kwargs = dict(time=40, illumination=50, laser_pulse_width=2)
    sim_1 = simulator_1.get_uniform_illumination(**kwargs)
    sim_2 = simulator_2.get_uniform_illumination(**kwargs)
    sim_3 = simulator_3.get_uniform_illumination(**kwargs)
    sim_4 = simulator_4.get_uniform_illumination(**kwargs)
    sim_5 = simulator_5.get_uniform_illumination(**kwargs)
    sim_6 = simulator_6.get_uniform_illumination(**kwargs)
    assert sim_1 != sim_2
    assert sim_2 != sim_3  # Generator has been progressed
    assert sim_3 != sim_4
    assert sim_4 == sim_5
    assert sim_5 != sim_6

    # get_cherenkov_shower
    shower_kwargs = dict(
        centroid_x=0.1,
        centroid_y=0,
        length=0.01,
        width=0.01,
        psi=0,
        time_gradient=1,
        time_intercept=20,
        intensity=100,
    )
    sim_1 = simulator_1.get_cherenkov_shower(**shower_kwargs)
    sim_2 = simulator_2.get_cherenkov_shower(**shower_kwargs)
    sim_3 = simulator_3.get_cherenkov_shower(**shower_kwargs)
    sim_4 = simulator_4.get_cherenkov_shower(**shower_kwargs)
    sim_5 = simulator_5.get_cherenkov_shower(**shower_kwargs)
    sim_6 = simulator_6.get_cherenkov_shower(**shower_kwargs)
    assert sim_1 != sim_2
    assert sim_2 != sim_3  # Generator has been progressed
    assert sim_3 != sim_4
    assert sim_4 == sim_5
    assert sim_5 != sim_6


def test_resample_photoelectron_charge():
    camera = Camera(
        mapping=SSTCameraMapping(n_pixels=2),
    )
    pe_0 = Photoelectrons(
        pixel=np.array([0, 1]),
        time=np.array([30, 40]),
        charge=np.array([1.0, 2.0]),
        metadata=dict(test=2)
    )
    source = PhotoelectronSource(camera=camera, seed=1)
    pe_1 = source.resample_photoelectron_charge(pe_0)
    pe_2 = source.resample_photoelectron_charge(pe_0)
    source = PhotoelectronSource(camera=camera, seed=2)
    pe_3 = source.resample_photoelectron_charge(pe_0)

    assert np.array_equal(pe_0.pixel, pe_1.pixel)
    assert np.array_equal(pe_1.pixel, pe_2.pixel)
    assert np.array_equal(pe_2.pixel, pe_3.pixel)
    assert np.array_equal(pe_0.time, pe_1.time)
    assert np.array_equal(pe_1.time, pe_2.time)
    assert np.array_equal(pe_2.time, pe_3.time)
    assert not np.array_equal(pe_0.charge, pe_1.charge)
    assert np.array_equal(pe_1.charge, pe_2.charge)
    assert not np.array_equal(pe_2.charge, pe_3.charge)
    assert pe_0.metadata == pe_1.metadata == pe_2.metadata == pe_3.metadata
