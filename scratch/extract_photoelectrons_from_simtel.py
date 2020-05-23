import argparse
from ctapipe.io import SimTelEventSource
from CHECLabPy.core.io import HDF5Writer
import pandas as pd
from tqdm import tqdm
import warnings


def process(path_list, output_path):
    df_list = []
    extracted_geometry = False
    with HDF5Writer(output_path) as writer:
        for path in tqdm(path_list, desc="Processing simtel files"):
            source = SimTelEventSource(path)

            if not extracted_geometry:
                subarray = source.subarray
                tel_id = list(subarray.tel.keys())[0]
                geometry = subarray.tel[tel_id].camera.geometry
                pixel = geometry.pix_id
                pix_x = geometry.pix_x
                pix_y = geometry.pix_y
                df_geom = pd.DataFrame(dict(
                    pixel=pixel,
                    pix_x=pix_x,
                    pix_y=pix_y,
                ))
                writer.write(geometry=df_geom)
                extracted_geometry = True

            for iev, event in tqdm(enumerate(source.file_)):
                photoelectrons = event['photoelectrons']
                mc_shower = event['mc_shower']
                mc_event = event['mc_event']

                energy = mc_shower["energy"]
                alt = mc_shower["altitude"]
                az = mc_shower["azimuth"]
                core_x = mc_event["xcore"]
                core_y = mc_event["ycore"]
                h_first_int = mc_shower["h_first_int"]
                x_max = mc_shower["xmax"]
                shower_primary_id = mc_shower["primary_id"]

                for tel, values in photoelectrons.items():
                    pixel_array = values['photoelectron_arrival_pixel']
                    time_array = values['photoelectron_arrival_time']

                    df_list.append(pd.DataFrame(dict(
                        iev=iev,
                        tel_id=tel+1,
                        pixel=[pixel_array],
                        time=[time_array],
                        energy=energy,
                        alt=alt,
                        az=az,
                        core_x=core_x,
                        core_y=core_y,
                        h_first_int=h_first_int,
                        x_max=x_max,
                        shower_primary_id=shower_primary_id
                    )))

        df = pd.concat(df_list, ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            writer.write(data=df)
            writer.add_metadata(n_rows=df.index.size)


def main():
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
    process(args.input_paths, args.output_path)


if __name__ == '__main__':
    main()
