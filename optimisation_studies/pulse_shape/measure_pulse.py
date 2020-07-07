from scipy.signal import find_peaks, peak_widths


def _extract_widths(pulse_y):
    peaks, _ = find_peaks(pulse_y)
    return peak_widths(pulse_y, peaks)


def extract_width(pulse_x, pulse_y):
    sample_width = pulse_x[1] - pulse_x[0]
    pulse_width = _extract_widths(pulse_y)[0][0] * sample_width

    undershoot_widths = _extract_widths(-pulse_y)
    if len(undershoot_widths[0]) == 0 :
        undershoot_width = 0
    else:
        undershoot_width = undershoot_widths[0][-1] * sample_width
    return pulse_width, undershoot_width


def extract_area(pulse_x, pulse_y):
    sample_width = pulse_x[1] - pulse_x[0]
    pulse_area = pulse_y[pulse_y > 0].sum() * sample_width
    undershoot_area = -1 * pulse_y[pulse_y < 0].sum() * sample_width
    return pulse_area, undershoot_area
