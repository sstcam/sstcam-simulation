from sstcam_simulation.io.photoelectron_writer import PhotoelectronWriter
from sstcam_simulation.io.photoelectron_reader import PhotoelectronReader
from sstcam_simulation.photoelectrons import Photoelectrons
import tables
import numpy as np
import pytest


@pytest.fixture(scope='function')
def pe_file(tmp_path):
    path = str(tmp_path / "test.h5")
    class EventTable(tables.IsDescription):
        test = tables.Float64Col()
        test2 = tables.Float64Col()

    random = np.random.RandomState(1)

    pe_0 = Photoelectrons(
        pixel=np.full(1, 0),
        time=np.full(1, 1.1),
        charge=np.full(1, 1.1),
        metadata=dict(test=3.4, test2=3.5)
    )
    pe_1 = Photoelectrons(
        pixel=random.randint(0, 2000, 1000000),
        time=random.random(1000000),
        charge=random.random(1000000),
        metadata=dict(test=2.4, test2=2.1)
    )
    pe_list = [pe_0, pe_1]

    with PhotoelectronWriter(path, EventTable) as writer:
        writer.append(pe_0)
        writer.append(pe_1)

    return path, pe_list


def test_init(pe_file):
    path, pe_list = pe_file
    reader = PhotoelectronReader(path)
    assert reader._file.isopen
    assert len(reader) == 2
    reader.close()
    assert not reader._file.isopen


def test_context(pe_file):
    path, pe_list = pe_file
    with PhotoelectronReader(path) as reader:
        file = reader._file
        assert file.isopen
        for i, pe in enumerate(reader):
            assert pe == pe_list[i]
    assert not file.isopen


def test_getitem(pe_file):
    path, pe_list = pe_file
    with PhotoelectronReader(path) as reader:
        assert reader[0] == pe_list[0]
        assert reader[1] == pe_list[1]


def test_random(pe_file):
    path, pe_list = pe_file
    pe = []
    with PhotoelectronReader(path) as reader:
        for _ in range(10):
            pe.append(len(reader.random_event()))
    assert np.array_equal(np.unique(pe), np.unique([len(p) for p in pe_list]))


def test_random_poisson_fluctuate(pe_file):
    path, pe_list = pe_file
    with PhotoelectronReader(path) as reader:
        pe1 = len(reader.random_event(poisson_fluctuate=True))

    pe = []
    for _ in range(10):
        with PhotoelectronReader(path) as reader:
            pe.append(len(reader.random_event(poisson_fluctuate=True)))

    assert not (pe1 == np.array(pe)).all()


def test_random_seed(pe_file):
    path, pe_list = pe_file
    with PhotoelectronReader(path, seed=1) as reader:
        pe1 = reader.random_event()

    for _ in range(10):
        with PhotoelectronReader(path, seed=1) as reader:
            assert pe1 == reader.random_event()

    with PhotoelectronReader(path, seed=2) as reader:
        assert pe1 != reader.random_event()

    with PhotoelectronReader(path, seed=1) as reader:
        pe1 = reader.random_event(poisson_fluctuate=True)

    with PhotoelectronReader(path, seed=1) as reader:
        pe2 = reader.random_event(poisson_fluctuate=True)

    with PhotoelectronReader(path, seed=2) as reader:
        pe3 = reader.random_event(poisson_fluctuate=True)

    assert pe1 == pe2
    assert pe2 != pe3
