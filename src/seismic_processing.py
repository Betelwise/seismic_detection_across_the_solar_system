# src/seismic_processing.py
import numpy as np
from obspy import read
from scipy import signal
from obspy.signal.trigger import classic_sta_lta, trigger_onset

def load_mseed_file(filepath):
    """Loads a single mseed file."""
    try:
        st = read(filepath)
        return st
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def preprocess_trace(trace, min_freq, max_freq, do_resample=False, resample_rate=None, clip_std_factor=None):
    """Applies bandpass filter, optional resampling, clipping, and normalization."""
    tr_filt = trace.copy()
    tr_filt.filter('bandpass', freqmin=min_freq, freqmax=max_freq, zerophase=True) # Added zerophase

    if do_resample and resample_rate:
        tr_filt.resample(resample_rate)

    tr_data = tr_filt.data

    if clip_std_factor:
        stddev = np.std(tr_data)
        threshold = clip_std_factor * stddev
        tr_data[np.abs(tr_data) > threshold] = np.sign(tr_data[np.abs(tr_data) > threshold]) * threshold # Clip instead of setting to stddev

    # Normalize between -1 and 1
    if np.ptp(tr_data) > 0: # Avoid division by zero if data is flat
        tr_data_normalized = 2 * (tr_data - np.min(tr_data)) / np.ptp(tr_data) - 1
    else:
        tr_data_normalized = np.zeros_like(tr_data)

    tr_filt.data = tr_data_normalized # Update trace data
    return tr_filt


def calculate_best_freq_range_by_pwr(trace, low_freq_point, high_freq_point, freq_window, num_top_powers=50):
    """
    Calculates the 'best' frequency band based on the average of top power values in spectrograms.
    """
    sampling_rate = trace.stats.sampling_rate
    freq_ranges = [
        (start, start + freq_window)
        for start in np.arange(low_freq_point, high_freq_point - freq_window + 0.01, 0.1) # Ensure high_freq_point is covered
    ]
    max_power_in_ranges = []

    for f_min, f_max in freq_ranges:
        st_filt_temp = trace.copy() # Use the original trace for each filter iteration
        st_filt_temp.filter('bandpass', freqmin=f_min, freqmax=f_max, zerophase=True)
        tr_filt_data = st_filt_temp.data

        if len(tr_filt_data) < 256: # Ensure enough data for spectrogram window
            print(f"Warning: Not enough data points ({len(tr_filt_data)}) for spectrogram in range {f_min}-{f_max}. Skipping.")
            continue

        # Adjust nperseg if trace is too short
        nperseg = min(256, len(tr_filt_data))
        noverlap = nperseg // 2

        fu, tu, sxxu = signal.spectrogram(tr_filt_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        if sxxu.size == 0: # Spectrogram might be empty if data is too short or all zeros
            print(f"Warning: Spectrogram is empty for range {f_min}-{f_max}. Skipping.")
            continue
        sxxu_dB = 10 * np.log10(sxxu + 1e-10) # Add small epsilon to avoid log(0)

        power_values_flat = sxxu_dB.flatten()
        top_power_values = np.sort(power_values_flat)[-num_top_powers:]
        avg_top_power = np.mean(top_power_values)
        max_power_in_ranges.append((f_min, f_max, avg_top_power))

    if not max_power_in_ranges:
        print("Warning: No valid frequency ranges found by power. Returning default.")
        return low_freq_point, low_freq_point + freq_window # Default if no ranges work

    best_range = max(max_power_in_ranges, key=lambda x: x[2])
    # print(f"Best frequency range by power: {best_range[0]:.1f}-{best_range[1]:.1f} Hz with avg power {best_range[2]:.3f}")
    return best_range[0], best_range[1]


def detect_triggers_sta_lta(trace_data, sampling_rate, sta_len_sec, lta_len_sec, thr_on, thr_off_factor=1.0):
    """
    Detects event triggers using STA/LTA.
    `trace_data` should be the processed (filtered, normalized) waveform data (e.g., absolute values).
    `sta_len_sec`, `lta_len_sec` are STA/LTA window lengths in seconds.
    """
    sta_samples = int(sta_len_sec * sampling_rate)
    lta_samples = int(lta_len_sec * sampling_rate)

    # Ensure sta/lta samples are at least 1
    sta_samples = max(1, sta_samples)
    lta_samples = max(1, lta_samples)

    # STA/LTA works best on positive data, like an envelope or absolute values
    cft = classic_sta_lta(trace_data, sta_samples, lta_samples)
    
    # Adjust thr_off relative to the mean of the CFT or a fixed low value
    # thr_off = np.mean(cft) * thr_off_factor # Or a fixed small value like 0.5 or 1.0
    # A more robust thr_off might be a fraction of thr_on or a value slightly above the noise level of CFT
    thr_off = thr_on / 2.0 # Example: half of the on-threshold

    on_off_indices = np.array(trigger_onset(cft, thr_on, thr_off))
    return cft, on_off_indices