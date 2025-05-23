import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from obspy import UTCDateTime # For time calculations

# Assuming your refactored functions are in src/
# Make sure src is in your PYTHONPATH or the script is in the project root
from src.config import get_event_type_from_path, get_settings
from src.seismic_processing import (
    load_mseed_file,
    preprocess_trace,
    calculate_best_freq_range_by_pwr,
    detect_triggers_sta_lta
)
from src.file_utils import save_event_catalog

# --- Configuration ---
MSEED_DATA_PATTERN = './old_src/data/lunar/training/data/S16_GradeB/*.mseed' # ADJUST THIS
OUTPUT_CATALOG_FILENAME = "labeled_events_mpl_v2.csv"
MAX_FILES_TO_PROCESS = None
ZOOM_WINDOW_SEC = 1*60*60 # Seconds before and after current event P-wave for zoom plot

# --- Global state for the interactive loop ---
current_label = None
fig_open = True

def on_key(event):
    """Handles key press events for labeling."""
    global current_label, fig_open
    # print(f"Key pressed: {event.key}") # Optional: for debugging
    if event.key == 'y':
        current_label = 1
        plt.close(event.canvas.figure)
    elif event.key == 'n':
        current_label = 0
        plt.close(event.canvas.figure)
    elif event.key == 's':
        current_label = -1
        plt.close(event.canvas.figure)
    elif event.key == 'q':
        current_label = -2
        fig_open = False
        plt.close(event.canvas.figure)
    elif event.key == 'escape':
        current_label = -3
        fig_open = False
        plt.close(event.canvas.figure)


def display_and_label_event(
    trace_processed,
    cft_data,
    all_triggers_in_file_samples, # List of all [start, end] triggers in this file
    current_trigger_idx_in_file, # Index of the current trigger we are labeling
    file_info,
    current_settings
):
    """
    Displays the event using Matplotlib (full trace and zoomed view) and waits for a key press.
    Returns the label.
    """
    global current_label, fig_open
    current_label = None
    fig_open = True

    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=False) # 3 plots now
    fig.canvas.mpl_connect('key_press_event', on_key)

    times = trace_processed.times() # Relative times for this trace
    waveform_data = trace_processed.data
    sampling_rate = trace_processed.stats.sampling_rate

    current_trigger_start_sample, current_trigger_end_sample = all_triggers_in_file_samples[current_trigger_idx_in_file]
    current_event_time_relative = current_trigger_start_sample / sampling_rate if sampling_rate > 0 else 0


    # --- Plot 1: Full Processed Waveform with all triggers ---
    axs[0].plot(times, waveform_data, label="Full Processed Waveform", color='C0', linewidth=0.7)
    axs[0].set_ylabel("Norm. Amplitude")
    axs[0].grid(True)
    axs[0].set_xlim(times[0], times[-1])

    for i, (start_s, end_s) in enumerate(all_triggers_in_file_samples):
        event_time_rel = start_s / sampling_rate if sampling_rate > 0 else 0
        line_color = 'red' if i == current_trigger_idx_in_file else 'green'
        line_style = '-' if i == current_trigger_idx_in_file else '--'
        line_width = 1.5 if i == current_trigger_idx_in_file else 1.0
        alpha_val = 1.0 if i == current_trigger_idx_in_file else 0.6
        axs[0].axvline(event_time_rel, color=line_color, linestyle=line_style, linewidth=line_width, alpha=alpha_val, label=f"Event {i+1}" if i == current_trigger_idx_in_file else None)
    axs[0].legend(loc="upper right", fontsize='small')


    # --- Plot 2: Zoomed-in Waveform around the current event ---
    zoom_start_time = max(0, current_event_time_relative - ZOOM_WINDOW_SEC)
    zoom_end_time = min(times[-1], current_event_time_relative + ZOOM_WINDOW_SEC)

    axs[1].plot(times, waveform_data, label=f"Zoomed Waveform (+/- {ZOOM_WINDOW_SEC}s)", color='C0')
    axs[1].set_ylabel("Norm. Amplitude")
    axs[1].grid(True)
    axs[1].set_xlim(zoom_start_time, zoom_end_time)

    # Add trigger lines to zoom plot, adjusting for visibility
    for i, (start_s, end_s) in enumerate(all_triggers_in_file_samples):
        event_time_rel = start_s / sampling_rate if sampling_rate > 0 else 0
        if zoom_start_time <= event_time_rel <= zoom_end_time: # Only plot if within zoom window
            line_color = 'red' if i == current_trigger_idx_in_file else 'green'
            line_style = '-' if i == current_trigger_idx_in_file else '--'
            line_width = 1.5 if i == current_trigger_idx_in_file else 1.0
            alpha_val = 1.0 if i == current_trigger_idx_in_file else 0.6
            axs[1].axvline(event_time_rel, color=line_color, linestyle=line_style, linewidth=line_width, alpha=alpha_val, label=f"Current Event {i+1}" if i == current_trigger_idx_in_file else f"Other Event {i+1}")
    axs[1].legend(loc="upper right", fontsize='small')


    # --- Plot 3: CFT (full view) ---
    if cft_data is not None:
        cft_times = np.linspace(0, times[-1] if len(times) > 0 else 0, len(cft_data), endpoint=False)
        axs[2].plot(cft_times, cft_data, label="STA/LTA CFT", color='orange')
        if 'thr_on' in current_settings:
             axs[2].axhline(current_settings['thr_on'], color='magenta', linestyle=':', label=f"Thr_on={current_settings['thr_on']:.2f}")
        axs[2].set_ylabel("CFT Value")
        axs[2].grid(True)
        axs[2].set_xlim(times[0], times[-1]) # Match full waveform x-axis
        axs[2].legend(loc="upper right", fontsize='small')

    axs[2].set_xlabel("Time (s) relative to trace start\nLabel: 'y' (Event), 'n' (Not), 's' (Skip), 'q' (Quit File), 'esc' (Quit All)")

    title_text = (f"File: {file_info} | Current Event (Red): {current_trigger_idx_in_file + 1} / {len(all_triggers_in_file_samples)}\n"
                  f"Relative Time of Current Event: {current_event_time_relative:.2f} s")
    fig.suptitle(title_text, fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjust for suptitle and xlabel
    plt.show()

    return current_label


def main():
    global fig_open

    mseed_files = sorted(glob.glob(MSEED_DATA_PATTERN))
    if not mseed_files:
        print(f"No mseed files found matching pattern: {MSEED_DATA_PATTERN}")
        return

    if MAX_FILES_TO_PROCESS is not None:
        mseed_files = mseed_files[:MAX_FILES_TO_PROCESS]

    all_labeled_events_info = [] # This will store dicts for the final catalog
    overall_event_index = 0 # Global index across all files

    print("Starting interactive labeling...")
    # ... (print instructions - same as before) ...

    for day_index, mseed_filepath in enumerate(mseed_files): # day_index assumes one file per day
        if not fig_open:
            print("Quitting all files.")
            break
        fig_open = True

        print(f"\nProcessing Day {day_index + 1} (File {day_index + 1}/{len(mseed_files)}): {os.path.basename(mseed_filepath)}")

        event_type = get_event_type_from_path(mseed_filepath)
        current_settings = get_settings(event_type)

        st = load_mseed_file(mseed_filepath)
        if not st or not st.traces:
            print(f"  Could not load or no traces in: {mseed_filepath}. Skipping.")
            continue

        original_trace = st[0].copy()
        trace_starttime_utc = original_trace.stats.starttime # UTCDateTime object

        try:
            low_f, high_f = calculate_best_freq_range_by_pwr(
                original_trace.copy(),
                current_settings['low_freq_point'],
                current_settings['high_freq_point'],
                current_settings['frequence_window']
            )
            print(f"  Using frequency range: {low_f:.1f} - {high_f:.1f} Hz")
        except Exception as e:
            print(f"  Error in calculate_best_freq_range_by_pwr: {e}. Using default range.")
            low_f = current_settings['low_freq_point']
            high_f = current_settings['low_freq_point'] + current_settings['frequence_window']

        trace_processed = preprocess_trace(
            original_trace.copy(),
            low_f, high_f,
            current_settings['resample'],
            current_settings.get('df_resample'),
            current_settings.get('clipping_std_factor')
        )

        trace_data_for_stalta = np.abs(trace_processed.data)
        sampling_rate = trace_processed.stats.sampling_rate
        if sampling_rate == 0: # Should not happen with valid data
            print("  Error: Sampling rate is zero. Skipping file.")
            continue

        cft, on_off_indices = detect_triggers_sta_lta(
            trace_data_for_stalta,
            sampling_rate,
            current_settings['sta_len'] / sampling_rate,
            current_settings['lta_len'] / sampling_rate,
            current_settings['thr_on']
        )

        all_triggers_in_file_samples = on_off_indices.tolist() # List of [start_sample, end_sample]

        min_duration_samples = int(2 * sampling_rate)
        filtered_triggers_in_file_samples = [
            trig for trig in all_triggers_in_file_samples
            if (trig[1] - trig[0]) > min_duration_samples
        ]

        if not filtered_triggers_in_file_samples:
            print("  No suitable triggers found in this file after filtering.")
            continue

        print(f"  Found {len(filtered_triggers_in_file_samples)} potential events to label in this file.")

        for event_index_in_day, current_trigger_samples in enumerate(filtered_triggers_in_file_samples):
            if not fig_open:
                break # from loop over triggers in current file

            overall_event_index += 1
            start_sample, _ = current_trigger_samples # We only need start for relative time here

            label = display_and_label_event(
                trace_processed,
                cft,
                filtered_triggers_in_file_samples, # Pass all triggers for highlighting
                event_index_in_day,                 # Index of the current one
                os.path.basename(mseed_filepath),
                current_settings
            )

            if label is None or label == -1:
                print(f"    Event {event_index_in_day + 1} (Overall {overall_event_index}) skipped.")
                overall_event_index -=1 # Decrement if skipped so next isn't misnumbered
                continue
            if label == -2:
                print(f"    Quitting file: {os.path.basename(mseed_filepath)}")
                overall_event_index -=1 # Decrement as this event wasn't fully processed for catalog
                break
            if label == -3:
                print(f"    Quitting all labeling.")
                overall_event_index -=1
                fig_open = False
                break

            # --- Catalog information ---
            relative_time_sec = start_sample / sampling_rate
            # absolute_time_utc = trace_starttime_utc + relative_time_sec # This is the P-wave arrival time

            event_info_for_catalog = {
                'filename': os.path.basename(mseed_filepath),
                'overall_event_id': overall_event_index, # Unique ID across all files
                'day_index': day_index + 1, # 1-based index for the file/day
                'event_index_in_day': event_index_in_day + 1, # 1-based index for event within its file
                'relative_time_sec': round(relative_time_sec, 3), # Relative to start of mseed file
                # 'absolute_time_utc': absolute_time_utc.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), # Optional
                'label': label # 0 or 1
            }
            all_labeled_events_info.append(event_info_for_catalog)
            print(f"    Event {event_index_in_day + 1} (Overall {overall_event_index}) labeled as: {'EVENT' if label == 1 else 'NOT EVENT'}")

    # After all files or quit
    if all_labeled_events_info:
        print(f"\nLabeling complete.")
        print(f"Collected {len(all_labeled_events_info)} labels.")
        output_dir = os.path.dirname(MSEED_DATA_PATTERN) if MSEED_DATA_PATTERN.endswith("*.mseed") else "."
        if not output_dir: output_dir = "." # Fallback if pattern is just "*.mseed"
        output_path = os.path.join(output_dir, OUTPUT_CATALOG_FILENAME)

        if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path) != '':
             os.makedirs(os.path.dirname(output_path), exist_ok=True)

        catalog_columns = [
            'filename', 'overall_event_id', 'day_index', 'event_index_in_day',
            'relative_time_sec', 'label' # Removed 'absolute_time_utc' for now, add back if needed
        ]
        save_event_catalog(all_labeled_events_info, output_path, catalog_columns)
    else:
        print("\nNo events were labeled.")


if __name__ == "__main__":
    main()