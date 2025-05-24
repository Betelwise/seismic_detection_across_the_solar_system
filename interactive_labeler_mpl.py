import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from scipy import signal # Needed for spectrogram
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
MSEED_DATA_PATTERN = './data/training/lunar/*.mseed' # ADJUST THIS
OUTPUT_CATALOG_FILENAME = "labeled_events_mpl_v3.csv" # Versioning output
MAX_FILES_TO_PROCESS = None
ZOOM_WINDOW_SEC = 2*60*60 # Seconds before and after current event P-wave for zoom plot
SPECTROGRAM_NPERSEG = 256 # Window size for STFT, adjust as needed
SPECTROGRAM_NOVERLAP_RATIO = 0.5 # Overlap ratio for STFT windows

# --- Global state for the interactive loop ---
current_label_or_action = 2 # Can be label (0,1), skip (-1), quit (-2,-3), or bulk action (e.g., 'bulk_0_10')
fig_open = True

def on_key(event):
    """Handles key press events for labeling."""
    global current_label_or_action, fig_open
    # print(f"Key pressed: {event.key}")
    if event.key == 'y':
        current_label_or_action = 1 # Is an event
        plt.close(event.canvas.figure)
    elif event.key == 'n':
        current_label_or_action = 0 # Not an event
        plt.close(event.canvas.figure)
    elif event.key == 's':
        current_label_or_action = -1 # Skip this event
        plt.close(event.canvas.figure)
    elif event.key == 'q':
        current_label_or_action = -2 # Quit current file
        fig_open = False
        plt.close(event.canvas.figure)
    elif event.key == 'escape':
        current_label_or_action = -3 # Quit all
        fig_open = False
        plt.close(event.canvas.figure)
    # Bulk labeling keys - using F-keys as an example
    elif event.key == 'f5': # Label next 5 as NOT EVENT
        current_label_or_action = 'bulk_0_5'
        plt.close(event.canvas.figure)
    elif event.key == 'f10': # Label next 10 as NOT EVENT
        current_label_or_action = 'bulk_0_10'
        plt.close(event.canvas.figure)
    elif event.key == 'f12': # Label next 20 as NOT EVENT
        current_label_or_action = 'bulk_0_20'
        plt.close(event.canvas.figure)


def display_and_label_event(
    trace_processed,
    cft_data,
    all_triggers_in_file_samples,
    current_trigger_idx_in_file,
    file_info,
    current_settings
):
    """
    Displays the event using Matplotlib and waits for a key press.
    Returns the label or action.
    """
    global current_label_or_action, fig_open
    current_label_or_action = None # Reset for this event
    fig_open = True # Reset for this plot

    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=False,
                        gridspec_kw={'height_ratios': [1, 1, 0.2]}) # Give spectrogram a bit more height
    fig.canvas.mpl_connect('key_press_event', on_key)

    times = trace_processed.times()
    waveform_data = trace_processed.data
    sampling_rate = trace_processed.stats.sampling_rate

    current_trigger_start_sample, _ = all_triggers_in_file_samples[current_trigger_idx_in_file]
    current_event_time_relative = current_trigger_start_sample / sampling_rate if sampling_rate > 0 else 0

    # --- Plot 1: Full Processed Waveform ---
    axs[0].plot(times, waveform_data, label="Full Processed Waveform", color='C0', linewidth=0.7)
    axs[0].set_ylabel("Norm. Amplitude")
    axs[0].grid(True)
    axs[0].set_xlim(times[0], times[-1])
    for i, (start_s, _) in enumerate(all_triggers_in_file_samples):
        event_time_rel = start_s / sampling_rate if sampling_rate > 0 else 0
        line_color = 'red' if i == current_trigger_idx_in_file else 'green'
        line_style = '-' if i == current_trigger_idx_in_file else '--'
        axs[0].axvline(event_time_rel, color=line_color, linestyle=line_style, linewidth=(1.5 if i == current_trigger_idx_in_file else 1.0), alpha=(1.0 if i == current_trigger_idx_in_file else 0.6), label=f"Event {i+1}" if i == current_trigger_idx_in_file else None)
    axs[0].legend(loc="upper right", fontsize='small')

    # --- Plot 2: Zoomed-in Waveform ---
    zoom_start_time = max(0, current_event_time_relative - ZOOM_WINDOW_SEC)
    zoom_end_time = min(times[-1], current_event_time_relative + ZOOM_WINDOW_SEC)
    axs[1].plot(times, waveform_data, label=f"Zoomed Waveform (+/- {ZOOM_WINDOW_SEC}s)", color='C0')
    axs[1].set_ylabel("Norm. Amplitude")
    axs[1].grid(True)
    axs[1].set_xlim(zoom_start_time, zoom_end_time)
    for i, (start_s, _) in enumerate(all_triggers_in_file_samples):
        event_time_rel = start_s / sampling_rate if sampling_rate > 0 else 0
        if zoom_start_time <= event_time_rel <= zoom_end_time:
            line_color = 'red' if i == current_trigger_idx_in_file else 'green'
            line_style = '-' if i == current_trigger_idx_in_file else '--'
            axs[1].axvline(event_time_rel, color=line_color, linestyle=line_style, linewidth=(1.5 if i == current_trigger_idx_in_file else 1.0), alpha=(1.0 if i == current_trigger_idx_in_file else 0.6))
    axs[1].legend(loc="upper right", fontsize='small')
    

    xlabel_text = (
        "Time (s) relative to trace start\n"
        "Label: 'y'(Evt), 'n'(Not), 's'(Skip), 'q'(Quit File), 'esc'(Quit All)\n"
        "Bulk Not Event: 'F5'(Next 5), 'F10'(Next 10), 'F12'(Next 20)"
    )
    axs[2].set_xlabel(xlabel_text)

    title_text = (f"File: {file_info} | Current Event (Red): {current_trigger_idx_in_file + 1} / {len(all_triggers_in_file_samples)}\n"
                  f"Relative Time of Current Event: {current_event_time_relative:.2f} s")
    fig.suptitle(title_text, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjusted for longer xlabel
    plt.show()

    return current_label_or_action


def add_event_to_catalog(filename, overall_id, day_idx, event_idx_in_day, rel_time_sec, label_val, catalog_list):
    """Helper function to create and append event dict to catalog list."""
    event_info = {
        'filename': filename,
        'overall_event_id': overall_id,
        'day_index': day_idx,
        'event_index_in_day': event_idx_in_day,
        'relative_time_sec': rel_time_sec,
        'label': label_val
    }
    catalog_list.append(event_info)

def main():
    global fig_open, current_label_or_action # current_label_or_action is key here

    mseed_files = sorted(glob.glob(MSEED_DATA_PATTERN))
    if not mseed_files:
        print(f"No mseed files found: {MSEED_DATA_PATTERN}")
        return

    if MAX_FILES_TO_PROCESS is not None:
        mseed_files = mseed_files[:MAX_FILES_TO_PROCESS]

    all_labeled_events_info = []
    overall_event_index_counter = 0 # This will strictly increment for each catalog entry

    print("Starting interactive labeling...")
    print("In the plot window, press:")
    print("  'y' to label as EVENT")
    print("  'n' to label as NOT EVENT")
    print("  's' to SKIP this detected event")
    print("  'q' to QUIT processing the current file")
    print("  'escape' to QUIT the entire labeling process")
    print("  'F5' to label this AND NEXT 4 events as NOT EVENT (total 5)")
    print("  'F10' to label this AND NEXT 9 events as NOT EVENT (total 10)")
    print("  'F12' to label this AND NEXT 19 events as NOT EVENT (total 20)")
    print("-" * 30)


    for day_idx, mseed_filepath in enumerate(mseed_files):
        if not fig_open: # Quit all
            print("Quitting all files.")
            break
        # Reset fig_open for the new file if 'q' was pressed for the previous file
        # but not 'escape' for quitting all.
        if current_label_or_action == -2: # if previous action was quit file
             fig_open = True
        current_label_or_action = None # Reset action at the start of each file processing


        print(f"\nProcessing Day {day_idx + 1} (File {day_idx + 1}/{len(mseed_files)}): {os.path.basename(mseed_filepath)}")

        event_type = get_event_type_from_path(mseed_filepath)
        current_settings = get_settings(event_type)
        st = load_mseed_file(mseed_filepath)
        if not st or not st.traces:
            print(f"  Could not load/no traces: {mseed_filepath}. Skipping.")
            continue

        original_trace = st[0].copy()
        # trace_starttime_utc = original_trace.stats.starttime

        try:
            low_f, high_f = calculate_best_freq_range_by_pwr(
                original_trace.copy(), current_settings['low_freq_point'],
                current_settings['high_freq_point'], current_settings['frequence_window']
            )
            print(f"  Using frequency range: {low_f:.1f} - {high_f:.1f} Hz")
        except Exception as e:
            print(f"  Error in freq calculation: {e}. Using default.")
            low_f = current_settings['low_freq_point']
            high_f = current_settings['low_freq_point'] + current_settings['frequence_window']

        trace_processed = preprocess_trace(
            original_trace.copy(), low_f, high_f, current_settings['resample'],
            current_settings.get('df_resample'), current_settings.get('clipping_std_factor')
        )
        sampling_rate = trace_processed.stats.sampling_rate
        if sampling_rate == 0:
            print("  Error: Sampling rate is zero. Skipping file.")
            continue

        cft, on_off_indices = detect_triggers_sta_lta(
            np.abs(trace_processed.data), sampling_rate,
            current_settings['sta_len'] / sampling_rate,
            current_settings['lta_len'] / sampling_rate,
            current_settings['thr_on']
        )
        all_triggers_in_file_samples = on_off_indices.tolist()
        min_duration_samples = int(2 * sampling_rate)
        filtered_triggers = [
            trig for trig in all_triggers_in_file_samples
            if (trig[1] - trig[0]) > min_duration_samples
        ]

        if not filtered_triggers:
            print("  No suitable triggers found in this file after filtering.")
            continue
        print(f"  Found {len(filtered_triggers)} potential events to label in this file.")

        # --- Loop through events in the current file ---
        event_idx_in_day_loop = 0
        while event_idx_in_day_loop < len(filtered_triggers):
            if not fig_open and current_label_or_action == -2 : # Quit current file
                break
            if not fig_open and current_label_or_action == -3 : # Quit all
                break


            current_trigger_samples = filtered_triggers[event_idx_in_day_loop]
            start_sample, _ = current_trigger_samples
            relative_time_sec = start_sample / sampling_rate

            # Display the current event and get action
            action = display_and_label_event(
                trace_processed, cft, filtered_triggers,
                event_idx_in_day_loop, os.path.basename(mseed_filepath), current_settings
            )

            if action is None or action == -1: # Skipped or closed window manually
                print(f"    Event {event_idx_in_day_loop + 1} skipped.")
                event_idx_in_day_loop += 1
                continue
            if action == -2: # Quit current file
                print(f"    Quitting file: {os.path.basename(mseed_filepath)}")
                break # Breaks from while loop for this file
            if action == -3: # Quit all
                print(f"    Quitting all labeling.")
                fig_open = False # ensure outer loop also exits
                break

            # Handle single event labeling
            if isinstance(action, int) and action in [0, 1]:
                overall_event_index_counter += 1
                add_event_to_catalog(
                    os.path.basename(mseed_filepath), overall_event_index_counter,
                    day_idx + 1, event_idx_in_day_loop + 1,
                    round(relative_time_sec, 3), action, all_labeled_events_info
                )
                print(f"    Event {event_idx_in_day_loop + 1} (Overall ID {overall_event_index_counter}) labeled as: {'EVENT' if action == 1 else 'NOT EVENT'}")
                event_idx_in_day_loop += 1

            # Handle bulk labeling
            elif isinstance(action, str) and action.startswith('bulk_0_'):
                try:
                    num_to_bulk_label = int(action.split('_')[-1])
                    print(f"    Bulk labeling next {num_to_bulk_label} events (including current) as NOT EVENT.")
                    for i in range(num_to_bulk_label):
                        if event_idx_in_day_loop < len(filtered_triggers):
                            overall_event_index_counter += 1
                            current_bulk_trigger = filtered_triggers[event_idx_in_day_loop]
                            current_bulk_start_sample, _ = current_bulk_trigger
                            current_bulk_rel_time = current_bulk_start_sample / sampling_rate

                            add_event_to_catalog(
                                os.path.basename(mseed_filepath), overall_event_index_counter,
                                day_idx + 1, event_idx_in_day_loop + 1,
                                round(current_bulk_rel_time, 3), 0, # Label 0 for "not event"
                                all_labeled_events_info
                            )
                            print(f"      Bulk: Event {event_idx_in_day_loop + 1} (Overall ID {overall_event_index_counter}) labeled as NOT EVENT.")
                            event_idx_in_day_loop += 1
                        else:
                            print("      Reached end of events in file during bulk labeling.")
                            break # Break from bulk labeling loop
                    # After bulk, the main while loop will continue from the new event_idx_in_day_loop
                except ValueError:
                    print(f"    Error parsing bulk action: {action}. Skipping event.")
                    event_idx_in_day_loop += 1 # Move to next to avoid infinite loop
            else: # Should not happen with current on_key
                print(f"    Unknown action: {action}. Skipping event.")
                event_idx_in_day_loop += 1


    # After all files or quit
    if all_labeled_events_info:
        print(f"\nLabeling complete. Collected {len(all_labeled_events_info)} labels.")
        output_dir = os.path.dirname(MSEED_DATA_PATTERN) if MSEED_DATA_PATTERN.endswith("*.mseed") else "."
        if not output_dir: output_dir = "."
        output_path = os.path.join(output_dir, OUTPUT_CATALOG_FILENAME)
        if not os.path.exists(os.path.dirname(output_path)) and os.path.dirname(output_path) != '':
             os.makedirs(os.path.dirname(output_path), exist_ok=True)

        catalog_columns = ['filename', 'overall_event_id', 'day_index', 'event_index_in_day', 'relative_time_sec', 'label']
        save_event_catalog(all_labeled_events_info, output_path, catalog_columns)
    else:
        print("\nNo events were labeled.")

if __name__ == "__main__":
    main()