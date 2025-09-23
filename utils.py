import numpy as np
from scipy.signal import butter, filtfilt
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes

def tanh_normalize(data, scale, offset):
    return np.tanh(scale * (data + offset))


def smooth(current_value, target_value, weight):
    current_value = (1.0 - weight) * current_value + weight * target_value
    return current_value


def map2dto1d(x, y, n):
    return x * n + y


def compute_snr(original_signal, filtered_signal):
    signal_power = np.var(filtered_signal)
    noise_power = np.var(original_signal - filtered_signal)
    snr = 10 * np.log10(signal_power / noise_power)
    return np.round(snr, 4)


## artifact detection inspired by openbci algorithm
## default threshold is 100 uV, absolute difference
## https://openbci.com/community/automated-eye-blink-detection-online-2/
def get_artifact_mask(data, sampling_rate, threshold=100, p_ratio=0.2, is_absolute=True):
    b, a = butter(2, 10 / (sampling_rate / 2), btype='low')  # 10 Hz lowpass filter
    
    # lowpass filter to blink range
    filtered = filtfilt(b, a, data)

    # create a mean rolling filtered copy
    filt_copy = np.copy(filtered)
    period = int(sampling_rate * p_ratio)
    for i in range(len(filt_copy)):
        DataFilter.perform_rolling_filter(filt_copy[i], period, AggOperations.MEAN)

    # find difference between original and rolling filtered
    diff = filtered - filt_copy
    
    # absolute difference if specified
    if is_absolute:
        diff = np.abs(diff)
    
    # create mask where 1 if artifact detected
    mask = diff > threshold

    return mask