
# Unit: nanosecond.
# Constant to simplify sampling and reference pulse definition.
SAMPLE_WIDTH = 1

# Sub-sampling division to emulate continuous readout for reference pulse and
# trigger line.
CONTINUOUS_SAMPLE_DIVISION = 10

# Unit: nanosecond.
# Defines sub-sampling to emulate continuous readout for reference pulse and trigger line.
CONTINUOUS_SAMPLE_WIDTH = SAMPLE_WIDTH / CONTINUOUS_SAMPLE_DIVISION
