from sstcam_simulation.io.photoelectron_writer import PhotoelectronWriter
from sstcam_simulation.io.photoelectron_reader import PhotoelectronReader
from sstcam_simulation.photoelectrons import Photoelectrons
import tables
import numpy as np


def test_photoelectron_reader(tmp_path):
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

    reader = PhotoelectronReader(path)
    assert reader._file.isopen
    reader.close()
    assert not reader._file.isopen

    with PhotoelectronReader(path) as reader:
        file = reader._file
        assert file.isopen
        for i, pe in enumerate(reader):
            assert pe == pe_list[i]
    assert not file.isopen
