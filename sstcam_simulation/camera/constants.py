__all__ = [
    "WAVEFORM_SAMPLE_WIDTH",
    "CONTINUOUS_READOUT_SAMPLE_DIVISION",
    "CONTINUOUS_READOUT_SAMPLE_WIDTH",
]

# Unit: nanosecond.
# Constant to simplify sampling and reference pulse definition.
WAVEFORM_SAMPLE_WIDTH = 1

# Sub-sampling division to emulate continuous readout for reference pulse and
# trigger line.
CONTINUOUS_READOUT_SAMPLE_DIVISION = 5

# Unit: nanosecond.
# Defines sub-sampling to emulate continuous readout for reference pulse and trigger line.
CONTINUOUS_READOUT_SAMPLE_WIDTH = WAVEFORM_SAMPLE_WIDTH / CONTINUOUS_READOUT_SAMPLE_DIVISION
