import argparse
from eventio import SimTelFile
import tables
from tqdm import tqdm


FILTERS = tables.Filters(
    complevel=5,  # compression medium, tradeoff between speed and compression
    complib="blosc:zstd",  # use modern zstd algorithm
    fletcher32=True,  # add checksums to data chunks
)


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


def add_telescope_table(h5_file, simtel_file):
    telescope_descriptions = simtel_file.telescope_descriptions
    tel_id = list(telescope_descriptions.keys())[0]
    camera_settings = telescope_descriptions[tel_id]['camera_settings']
    pixel_x = camera_settings['pixel_x']
    pixel_y = camera_settings['pixel_y']

    group = h5_file.create_group(h5_file.root, "Geometry", "Camera Geometry")
    h5_file.create_array(group, 'pixel_x', pixel_x, "Pixel X Coordinates")
    h5_file.create_array(group, 'pixel_y', pixel_y, "Pixel Y Coordinates")


def extract_events(path, table, pe_time, pe_pixel):
    i = 0
    with SimTelFile(path) as simtel_file:
        for iev, event in tqdm(enumerate(simtel_file.iter_mc_events())):
            if 'photoelectrons' not in event:
                continue

            photoelectrons = event['photoelectrons']
            mc_shower = event['mc_shower']
            mc_event = event['mc_event']

            for tel, values in photoelectrons.items():
                if values['time'].size < 100:
                    continue

                pe_time.append(values['time'])
                pe_pixel.append(values['pixel_id'])

                event = table.row
                event['event_index'] = iev
                event['telescope_id'] = tel + 1
                event['n_photoelectrons'] = values['time'].size
                event['energy'] = mc_shower["energy"]
                event['alt'] = mc_shower["altitude"]
                event['az'] = mc_shower["azimuth"]
                event['core_x'] = mc_event["xcore"]
                event['core_y'] = mc_event["ycore"]
                event['h_first_int'] = mc_shower["h_first_int"]
                event['x_max'] = mc_shower["xmax"]
                event['shower_primary_id'] = mc_shower["primary_id"]
                event.append()
                i += 1
            if i > 3:
                break


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
    args = parser.parse_args()
    path_list = args.input_paths
    output_path = args.output_path

    with tables.File(output_path, mode='w', filters=FILTERS) as h5_file:

        # Extract geometry
        with SimTelFile(path_list[0]) as simtel_file:
            add_telescope_table(h5_file, simtel_file)

        # Create event table
        group = h5_file.create_group(h5_file.root, "Data", "Data")
        table = h5_file.create_table(group, "event", EventTable, "Simtel Events")
        pe_time = h5_file.create_vlarray(
            group,
            "photoelectron_arrival_time",
            tables.Float64Atom(shape=()),
            "Arrival time of the photoelectrons"
        )
        pe_pixel = h5_file.create_vlarray(
            group,
            "photoelectron_arrival_pixel",
            tables.UInt16Atom(shape=()),
            "Pixel hit by the photoelectron"
        )

        for path in tqdm(path_list, desc="Processing simtel files"):
            extract_events(path, table, pe_time, pe_pixel)
            h5_file.flush()


if __name__ == '__main__':
    main()
