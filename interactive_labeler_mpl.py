import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob

# Assuming your refactored functions are in src/
# Make sure src is in your PYTHONPATH or the script is in the project root
from src.config import get_event_type_from_path, get_settings
from src.seismic_processing import (
    load_mseed_file,
    preprocess_trace,
    calculate_best_freq_range_by_pwr, # Or your preferred method
    detect_triggers_sta_lta
)
from src.file_utils import save_event_catalog # We'll use this

# --- Configuration ---
MSEED_DATA_PATTERN = './old_src/data/lunar/training/data/S12_GradeA/*.mseed' # ADJUST THIS
# MSEED_DATA_PATTERN = './data/mars/training/data/*.mseed'
OUTPUT_CATALOG_FILENAME = "labeled_events_mpl.csv"
MAX_FILES_TO_PROCESS = None # Set to a number to limit, or None for all

# --- Global state for the interactive loop (less ideal, but simpler for this case) ---
current_label = None
fig_open = True # Flag to control the plot loop

def on_key(event):
    """Handles key press events for labeling."""
    global current_label, fig_open
    print(f"Key pressed: {event.key}")
    if event.key == 'y':
        current_label = 1 # Is an event
        plt.close(event.canvas.figure)
    elif event.key == 'n':
        current_label = 0 # Not an event
        plt.close(event.canvas.figure)
    elif event.key == 's':
        current_label = -1 # Skip this event
        plt.close(event.canvas.figure)
    elif event.key == 'q':
        current_label = -2 # Quit current file (or all if you prefer)
        fig_open = False # Signal to exit outer loop if needed
        plt.close(event.canvas.figure)
    elif event.key == 'escape': # General quit
        current_label = -3
        fig_open = False
        plt.close(event.canvas.figure)


def display_and_label_event(trace_processed, cft_data, trigger_window_samples, file_info, event_num_in_file, total_events_in_file):
    """
    Displays the event using Matplotlib and waits for a key press.
    Returns the label.
    """
    global current_label, fig_open
    current_label = None # Reset label for this event
    fig_open = True # Reset for this plot

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.canvas.mpl_connect('key_press_event', on_key)

    times = trace_processed.times()
    waveform_data = trace_processed.data

    # Plot 1: Processed Waveform
    axs[0].plot(times, waveform_data, label="Processed Waveform")
    axs[0].set_ylabel("Normalized Amplitude")
    axs[0].grid(True)

    # Plot 2: CFT
    if cft_data is not None:
        # Ensure CFT times align if resampling happened
        cft_times = np.linspace(0, times[-1] if len(times) > 0 else 0, len(cft_data), endpoint=False)
        axs[1].plot(cft_times, cft_data, label="STA/LTA CFT", color='orange')
        # You might want to fetch thr_on from settings to plot it
        # axs[1].axhline(current_settings['thr_on'], color='red', linestyle='--', label=f"Thr_on")
        axs[1].set_ylabel("CFT Value")
        axs[1].grid(True)

    # Highlight current trigger window
    start_idx, end_idx = trigger_window_samples
    if 0 <= start_idx < len(times) and 0 <= end_idx < len(times) and start_idx < end_idx:
        start_time = times[start_idx]
        end_time = times[end_idx]
        axs[0].axvspan(start_time, end_time, color='yellow', alpha=0.4, label="Current Event Window")
        if cft_data is not None:
            axs[1].axvspan(start_time, end_time, color='yellow', alpha=0.4)
    else:
        print(f"Warning: Invalid trigger indices {start_idx}-{end_idx} for plotting span.")


    axs[0].legend(loc="upper right")
    if cft_data is not None:
        axs[1].legend(loc="upper right")
    axs[1].set_xlabel("Time (s)\nLabel: 'y' (Event), 'n' (Not Event), 's' (Skip), 'q' (Quit File), 'esc' (Quit All)")

    title_text = f"File: {file_info}\nEvent {event_num_in_file}/{total_events_in_file}"
    fig.suptitle(title_text, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show() # This blocks until the window is closed (manually or by on_key)

    return current_label


def main():
    global fig_open # To allow quitting all files

    mseed_files = sorted(glob.glob(MSEED_DATA_PATTERN))
    if not mseed_files:
        print(f"No mseed files found matching pattern: {MSEED_DATA_PATTERN}")
        return

    if MAX_FILES_TO_PROCESS is not None:
        mseed_files = mseed_files[:MAX_FILES_TO_PROCESS]

    all_labeled_events = []
    processed_file_count = 0

    print("Starting interactive labeling...")
    print("In the plot window, press:")
    print("  'y' to label as EVENT")
    print("  'n' to label as NOT EVENT")
    print("  's' to SKIP this detected event")
    print("  'q' to QUIT processing the current file and move to the next")
    print("  'escape' to QUIT the entire labeling process")
    print("-" * 30)

    for file_idx, mseed_filepath in enumerate(mseed_files):
        if not fig_open: # If 'escape' was pressed
            print("Quitting all files.")
            break
        fig_open = True # Reset for the new file if 'q' was pressed previously

        print(f"\nProcessing file {file_idx + 1}/{len(mseed_files)}: {os.path.basename(mseed_filepath)}")
        processed_file_count += 1

        event_type = get_event_type_from_path(mseed_filepath)
        current_settings = get_settings(event_type)

        st = load_mseed_file(mseed_filepath)
        if not st or not st.traces:
            print(f"Could not load or no traces in: {mseed_filepath}. Skipping.")
            continue

        original_trace = st[0].copy() # Assuming first trace

        # 1. Determine best frequency (or use fixed from settings)
        # For simplicity, let's assume calculate_best_freq_range_by_pwr works fine.
        # You might want to add error handling or fallbacks here.
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


        # 2. Preprocess
        trace_processed = preprocess_trace(
            original_trace.copy(),
            low_f, high_f,
            current_settings['resample'],
            current_settings.get('df_resample'),
            current_settings.get('clipping_std_factor')
        )

        # 3. STA/LTA
        trace_data_for_stalta = np.abs(trace_processed.data) # STA/LTA on absolute values
        sampling_rate = trace_processed.stats.sampling_rate
        cft, on_off_indices = detect_triggers_sta_lta(
            trace_data_for_stalta,
            sampling_rate,
            current_settings['sta_len'] / sampling_rate if sampling_rate > 0 else current_settings['sta_len'], # STA in sec
            current_settings['lta_len'] / sampling_rate if sampling_rate > 0 else current_settings['lta_len'], # LTA in sec
            current_settings['thr_on']
        )

        detected_triggers = on_off_indices.tolist()

        # Optional: Filter triggers by duration, etc.
        min_duration_samples = int(2 * sampling_rate if sampling_rate > 0 else 10) # e.g., 2 seconds
        filtered_triggers = [
            trig for trig in detected_triggers
            if (trig[1] - trig[0]) > min_duration_samples
        ]

        if not filtered_triggers:
            print("  No suitable triggers found in this file after filtering.")
            continue

        print(f"  Found {len(filtered_triggers)} potential events to label.")

        for event_idx, trigger_samples in enumerate(filtered_triggers):
            if not fig_open: # If 'q' or 'escape' was pressed in the previous event's plot
                break

            start_sample, end_sample = trigger_samples
            label = display_and_label_event(
                trace_processed,
                cft,
                (start_sample, end_sample),
                os.path.basename(mseed_filepath),
                event_idx + 1,
                len(filtered_triggers)
            )

            if label is None or label == -1: # Skipped or closed window manually without key
                print(f"    Event {event_idx + 1} skipped.")
                continue
            if label == -2: # 'q' pressed: quit current file
                print(f"    Quitting file: {os.path.basename(mseed_filepath)}")
                break # Breaks from loop over triggers in current file
            if label == -3: # 'escape' pressed: quit all
                print(f"    Quitting all labeling.")
                fig_open = False # Ensure outer loop also exits
                break

            # Record the labeled event
            start_time_abs = trace_processed.stats.starttime + (start_sample / sampling_rate if sampling_rate > 0 else 0)
            end_time_abs = trace_processed.stats.starttime + (end_sample / sampling_rate if sampling_rate > 0 else 0)

            event_data = {
                'filename': os.path.basename(mseed_filepath),
                'event_number_in_file': event_idx + 1,
                'trigger_start_sample': start_sample,
                'trigger_end_sample': end_sample,
                'trigger_start_time_relative_sec': round(start_sample / sampling_rate if sampling_rate > 0 else 0, 3),
                'trigger_end_time_relative_sec': round(end_sample / sampling_rate if sampling_rate > 0 else 0, 3),
                'trigger_start_time_utc': start_time_abs.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if sampling_rate > 0 else "N/A",
                'trigger_end_time_utc': end_time_abs.strftime('%Y-%m-%dT%H:%M:%S.%fZ') if sampling_rate > 0 else "N/A",
                'filter_low_hz': round(low_f, 2),
                'filter_high_hz': round(high_f, 2),
                'label': label # 0 or 1
            }
            all_labeled_events.append(event_data)
            print(f"    Event {event_idx + 1} labeled as: {'EVENT' if label == 1 else 'NOT EVENT'}")

    # After all files or quit
    if all_labeled_events:
        print(f"\nLabeling complete. Processed {processed_file_count} files.")
        print(f"Collected {len(all_labeled_events)} labels.")
        output_path = os.path.join(os.path.dirname(MSEED_DATA_PATTERN), OUTPUT_CATALOG_FILENAME) # Save near data
        if not os.path.exists(os.path.dirname(output_path)):
             os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Define columns for the CSV file
        columns = [
            'filename', 'event_number_in_file',
            'trigger_start_sample', 'trigger_end_sample',
            'trigger_start_time_relative_sec', 'trigger_end_time_relative_sec',
            'trigger_start_time_utc', 'trigger_end_time_utc',
            'filter_low_hz', 'filter_high_hz', 'label'
        ]
        save_event_catalog(all_labeled_events, output_path, columns)
    else:
        print("\nNo events were labeled.")

if __name__ == "__main__":
    # Important: Ensure Matplotlib interactive mode is suitable for your environment.
    # For some backends/OS, plt.show() might behave differently.
    # You might need to explicitly set a backend:
    # import matplotlib
    # matplotlib.use('TkAgg') # Or 'Qt5Agg', etc.
    main()