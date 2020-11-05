from sstcam_simulation.io.photoelectron_writer import PhotoelectronWriter
from sstcam_simulation.photoelectrons import Photoelectrons
import tables
import numpy as np
import pytest


def test_photoelectron_writer(tmp_path):
    path = str(tmp_path / "test.h5")
    class EventTable(tables.IsDescription):
        test = tables.Float64Col()

    writer = PhotoelectronWriter(path, EventTable)
    assert writer._file.isopen
    assert writer._file.root.data.photoelectron_arrival_pixel.nrows == 0
    assert writer._file.root.data.photoelectron_arrival_time.nrows == 0
    assert writer._file.root.data.photoelectron_measured_charge.nrows == 0
    assert writer._file.root.data.event_metadata.colnames == ["test"]

    writer.close()
    assert not writer._file.isopen

    class EventTable(tables.IsDescription):
        test2 = tables.Float64Col()

    with PhotoelectronWriter(path, EventTable) as writer:
        file = writer._file
        assert file.isopen
        assert writer._file.root.data.event_metadata.colnames == ["test2"]
    assert not file.isopen


def test_photoelectron_writer_append(tmp_path):
    path = str(tmp_path / "test.h5")
    class EventTable(tables.IsDescription):
        test = tables.Float64Col()

    random = np.random.RandomState(1)

    pe_0 = Photoelectrons(
        pixel=np.full(1, 0),
        time=np.full(1, 1.1),
        charge=np.full(1, 1.1),
        metadata=dict(test=3.4)
    )
    pe_1 = Photoelectrons(
        pixel=random.randint(0, 2000, 1000000),
        time=random.random(1000000),
        charge=random.random(1000000),
        metadata=dict(test=2.4)
    )

    def check_contents(pixel, time, charge, table):
        assert len(pixel) == 2
        assert np.array_equal(pixel[0], pe_0.pixel)
        assert np.array_equal(pixel[1], pe_1.pixel)
        assert len(time) == 2
        assert np.array_equal(time[0], pe_0.time)
        assert np.array_equal(time[1], pe_1.time)
        assert len(charge) == 2
        assert np.array_equal(charge[0], pe_0.charge)
        assert np.array_equal(charge[1], pe_1.charge)
        assert len(table) == 2
        assert np.array_equal(table[0][0], pe_0.metadata['test'])
        assert np.array_equal(table[1][0], pe_1.metadata['test'])

    with PhotoelectronWriter(path, EventTable) as writer:
        assert writer._file.root.data.photoelectron_arrival_pixel.nrows == 0
        assert writer._file.root.data.photoelectron_arrival_time.nrows == 0
        assert writer._file.root.data.photoelectron_measured_charge.nrows == 0
        assert writer._file.root.data.event_metadata.nrows == 0
        writer.append(pe_0)
        writer.append(pe_1)
        writer._file.flush()

        assert writer._file.root.data.photoelectron_arrival_pixel.nrows == 2
        assert writer._file.root.data.photoelectron_arrival_time.nrows == 2
        assert writer._file.root.data.photoelectron_measured_charge.nrows == 2
        assert writer._file.root.data.event_metadata.nrows == 2

        pixel = writer._file.root.data.photoelectron_arrival_pixel.read()
        time = writer._file.root.data.photoelectron_arrival_time.read()
        charge = writer._file.root.data.photoelectron_measured_charge.read()
        table = writer._file.root.data.event_metadata.read()

        check_contents(pixel, time, charge, table)

    with tables.File(path, mode='r') as file:
        pixel = file.root.data.photoelectron_arrival_pixel.read()
        time = file.root.data.photoelectron_arrival_time.read()
        charge = file.root.data.photoelectron_measured_charge.read()
        table = file.root.data.event_metadata.read()

        check_contents(pixel, time, charge, table)

    with PhotoelectronWriter(path, EventTable) as writer:
        with pytest.raises(KeyError):
            writer.append(Photoelectrons(
                pixel=np.full(1, 0),
                time=np.full(1, 1.1),
                charge=np.full(1, 1.1),
                metadata=dict(test=3.5, test2=3.3)
            ))
        writer._file.flush()
        assert writer._file.root.data.photoelectron_arrival_pixel.nrows == 0
        assert writer._file.root.data.event_metadata.nrows == 0

    with PhotoelectronWriter(path, EventTable) as writer:
        writer.append(Photoelectrons(
            pixel=np.full(1, 0),
            time=np.full(1, 1.1),
            charge=np.full(1, 1.1),
        ))
        writer._file.flush()
        assert writer._file.root.data.photoelectron_arrival_pixel.nrows == 1
        assert writer._file.root.data.event_metadata.nrows == 1
        assert writer._file.root.data.event_metadata[0]['test'] == 0
