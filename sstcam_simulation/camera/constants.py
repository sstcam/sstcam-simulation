
# Unit: nanosecond.
# Constant to simplify sampling and reference pulse definition.
SAMPLE_WIDTH = 1

# Unit: nanosecond.
# Defines sub-sampling to emulate continuous readout for reference pulse and trigger line.
# Set to a tenth of the sample_width to simplify sampling process,
# and to be a satisfactory size for its purpose.
SUB_SAMPLE_WIDTH = SAMPLE_WIDTH / 10
