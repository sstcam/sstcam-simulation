from sstcam_simulation import SimtelReader, PhotoelectronWriter
import argparse
import tables
from tqdm import tqdm


class EventTable(tables.IsDescription):
    event_index = tables.UInt64Col()
    telescope_id = tables.UInt8Col()
    n_photoelectrons = tables.UInt64Col()
    energy = tables.Float64Col()
    alt = tables.Float64Col()
    az = tables.Float64Col()
    core_x = tables.Float64Col()
    core_y = tables.Float64Col()
    h_first_int = tables.Float64Col()
    x_max = tables.Float64Col()
    shower_primary_id = tables.UInt8Col()


def main():
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(
        description="Extract the photoelectrons from a simtelarray file for "
                    "use in sstcam-simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-f', '--files', dest='input_paths', nargs='+',
        help='path to the simtelarray files'
    )
    parser.add_argument(
        '-o', '--output', dest='output_path', action='store',
        help='path to store the output HDF5 file'
    )
    parser.add_argument(
        '-n', '--n_events', dest='n_events', action='store',
        help='number of telescope events to store'
    )
    parser.add_argument(
        '-T', dest='only_tiggered_events', action='store_true',
        help='store only event that produced a telescope trigger in simtelarray'
    )
    args = parser.parse_args()
    path_list = args.input_paths
    output_path = args.output_path
    kwargs = dict(
        disable_remapping=False,
        only_triggered_events=args.only_triggered_events,
        n_events=args.n_events,
    )

    with PhotoelectronWriter(output_path, EventTable) as output:
        for path in tqdm(path_list, desc="Processing simtel files"):
            reader = SimtelReader(path, **kwargs)
            for pe in reader:
                output.append(pe)
            output.flush()


if __name__ == '__main__':
    main()
