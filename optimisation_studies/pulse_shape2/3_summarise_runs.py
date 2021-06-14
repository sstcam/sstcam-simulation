import argparse
import tables
import pandas as pd
from sstcam_simulation.performance.plot_lab_performance import ChargeResolution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', dest='h5_path', help='Path to the events file'
    )
    args = parser.parse_args()
    h5_path = args.h5_path

    print(f"Extracting data: {h5_path}")
    with tables.File(h5_path) as file:
        metadata = file.root.event.attrs
        first_row = file.root.event[0]
        data = file.root.event[:]

    print("Calculating performance")

    shower_primary_id = first_row['shower_primary_id']
    pulse_width = metadata.pulse_width
    trigger_threshold = metadata.trigger_threshold
    pedestal = metadata.pedestal
    pulse_area = metadata.pulse_area
    spectrum_average = metadata.spectrum_average

    measured_charge = data['measured_charge']
    calibrated_charge = (measured_charge - pedestal) / (spectrum_average * pulse_area)
    signal_pe = data['signal_pe']
    signal_charge = data['signal_charge'] / spectrum_average
    n_triggers = data['n_triggers']

    cr_extracted = ChargeResolution(mc_true=True)
    cr_extracted.add(0, signal_pe.ravel(), calibrated_charge.ravel())
    cr_extracted._amalgamate()
    cr_extracted_df = cr_extracted._df

    cr_signal = ChargeResolution(mc_true=True)
    cr_signal.add(0, signal_pe.ravel(), signal_charge.ravel())
    cr_signal._amalgamate()
    cr_signal_df = cr_signal._df

    image_amplitude = signal_pe.sum(1)
    triggered = n_triggers.sum(1) > 0
    triggers_df = pd.DataFrame(dict(image_amplitude=image_amplitude, triggered=triggered))

    metadata = dict(
        path=h5_path,
        shower_primary_id=shower_primary_id,
        pulse_width=pulse_width,
        trigger_threshold=trigger_threshold,
        pedestal=pedestal,
        pulse_area=pulse_area,
        spectrum_average=spectrum_average,
    )

    output_path = h5_path.replace("_events.h5", "_summary.h5")
    print(f"Storing output: {output_path}")
    with pd.HDFStore(output_path) as store:
        store['cr_extracted'] = cr_extracted_df
        store['cr_signal'] = cr_signal_df
        store['triggers'] = triggers_df
        store['metadata'] = pd.DataFrame(metadata, index=[0])


if __name__ == '__main__':
    main()
