import pandas as pd
import numpy as np
from obspy import read, UTCDateTime # UTCDateTime needed for slicing
# from obspy.signal.filter import bandpass # Not directly used if st.filter is preferred
import os
import glob

# --- Configuration ---
BASE_TRAINING_DIR = './data/training' # Root directory containing 'lunar', 'mars', etc.
CATALOG_FILENAME = 'catalog.csv'      # Name of the catalog file inside each subfolder
TRACE_NPA_DIR_NAME = 'trace_npa'      # Output folder for trace numpy arrays
AUX_NPA_DIR_NAME = 'aux_npa'          # Output folder for auxiliary numpy arrays

from src.config import get_event_type_from_path, get_settings
from src.seismic_processing import (
    load_mseed_file,
    preprocess_trace,
    calculate_best_freq_range_by_pwr,
    detect_triggers_sta_lta
)

# Define the fixed length for the data and the desired sampling rate for processing
# This fixed_length should correspond to 14 minutes of data AT THE TARGET SAMPLING RATE
TARGET_SAMPLING_RATE_HZ = 6.625  # Desired sampling rate for all processed traces (Hz)
WINDOW_DURATION_SEC = 14 * 60  # 14 minutes in seconds
FIXED_LENGTH_SAMPLES = int(TARGET_SAMPLING_RATE_HZ * WINDOW_DURATION_SEC) # Samples for 14 mins at TARGET_SAMPLING_RATE_HZ

TIME_BEFORE_EVENT_SEC = 4 * 60 # 4 minutes before the event for slicing
TIME_AFTER_EVENT_SEC = 10 * 60 # 10 minutes after the event for slicing

# --- Helper Function ---
def process_and_save_data(
        mseed_filepath, 
        relative_event_time_sec, 
        ):
    """
    Processes a single event from an mseed file and saves the trace and auxiliary data.
    Returns True if successful, False otherwise.
    """
    try:
        st = load_mseed_file(mseed_filepath)
        if not st:
            print(f"  Warning: Could not read or empty stream: {mseed_filepath}")
            return None, None # Indicate failure to process this event

        # Ensure single trace (or select the appropriate one if multiple)
        if len(st) > 1:
            print(f"  Warning: Multiple traces in {mseed_filepath}. Using the first one.")
        # tr = st[0].copy() # Work with a copy

        # # 0. Resample to target sampling rate BEFORE any other processing
        # # This ensures all snippets have consistent timing and length calculations
        # if tr.stats.sampling_rate != TARGET_SAMPLING_RATE_HZ:
        #     try:
        #         print(f"    Resampling {mseed_filepath} from {tr.stats.sampling_rate} Hz to {TARGET_SAMPLING_RATE_HZ} Hz.")
        #         tr.resample(TARGET_SAMPLING_RATE_HZ)
        #     except Exception as e:
        #         print(f"  Error resampling {mseed_filepath}: {e}. Skipping event.")
        #         return None, None
        
        # Recalculate relative_event_time_samples based on the new TARGET_SAMPLING_RATE_HZ
        # This is important if the original mseed had a different rate
        # relative_event_time_samples_at_target_sr = int(relative_event_time_sec * TARGET_SAMPLING_RATE_HZ)

        event_type = get_event_type_from_path(mseed_filepath)
        if event_type is None:
            print(f"  Warning: Could not determine event type from path: {mseed_filepath}. Skipping.")
            return None, None
        current_settings = get_settings(event_type)
        original_trace = st[0].copy()

        try:
            low_f, high_f = calculate_best_freq_range_by_pwr(
                original_trace.copy(), current_settings['low_freq_point'],
                current_settings['high_freq_point'], current_settings['frequence_window']
            )
        except Exception: # Fallback
            low_f = current_settings['low_freq_point']
            high_f = current_settings['low_freq_point'] + current_settings['frequence_window']

        tr_filtered = preprocess_trace(
            original_trace.copy(), low_f, high_f, current_settings['resample'],
            current_settings.get('df_resample'), current_settings.get('clipping_std_factor')
        )

        # 2. Define the slice window based on relative_event_time_sec
        # The times are relative to the START of the (potentially resampled) trace
        slice_start_sec_rel = relative_event_time_sec - TIME_BEFORE_EVENT_SEC
        slice_end_sec_rel = relative_event_time_sec + TIME_AFTER_EVENT_SEC

        # Ensure slice times are within trace bounds
        trace_duration_sec = tr_filtered.stats.npts / tr_filtered.stats.sampling_rate
        slice_start_sec_rel = max(0, slice_start_sec_rel)
        slice_end_sec_rel = min(trace_duration_sec, slice_end_sec_rel)

        if slice_start_sec_rel >= slice_end_sec_rel:
            print(f"  Warning: Invalid slice window for {mseed_filepath} (event at {relative_event_time_sec}s). "
                  f"Start: {slice_start_sec_rel}, End: {slice_end_sec_rel}. Skipping.")
            return None, None

        # Slice the filtered trace
        # Obspy's slice uses absolute UTCDateTime objects
        abs_trace_starttime = tr_filtered.stats.starttime # This is a UTCDateTime object
        abs_slice_starttime = abs_trace_starttime + slice_start_sec_rel
        abs_slice_endtime = abs_trace_starttime + slice_end_sec_rel
        
        tr_sliced = tr_filtered.slice(starttime=abs_slice_starttime, endtime=abs_slice_endtime)

        if not tr_sliced or not tr_sliced[0].data.any(): # Check if slice is empty or all zeros
            print(f"  Warning: Slice resulted in empty or zero trace for {mseed_filepath} (event at {relative_event_time_sec}s). Skipping.")
            return None, None
        
        sliced_data = tr_sliced[0].data.astype(np.float32) # Ensure float32 for consistency

        # 3. Normalize the sliced data (mean=0, std=1)
        mean_val = np.mean(sliced_data)
        std_val = np.std(sliced_data)
        if std_val == 0: # Avoid division by zero for flat traces
            sliced_data_normalized = np.zeros_like(sliced_data)
            print(f"  Warning: Std dev of sliced data is 0 for {mseed_filepath} (event at {relative_event_time_sec}s). Using zeros.")
        else:
            sliced_data_normalized = (sliced_data - mean_val) / std_val

        # 4. Pad or truncate to FIXED_LENGTH_SAMPLES
        current_len = len(sliced_data_normalized)
        if current_len > FIXED_LENGTH_SAMPLES:
            # Truncate: Take a window centered around the expected relative event position within the slice
            # Event was at TIME_BEFORE_EVENT_SEC into the slice_start_sec_rel
            # So, its position in the `sliced_data_normalized` array is roughly:
            event_pos_in_slice = int(TIME_BEFORE_EVENT_SEC * TARGET_SAMPLING_RATE_HZ)
            
            # Adjust event_pos_in_slice if slice_start_sec_rel was pushed to 0
            if slice_start_sec_rel == 0:
                 event_pos_in_slice = int(relative_event_time_sec * TARGET_SAMPLING_RATE_HZ)


            start_idx_truncate = max(0, event_pos_in_slice - FIXED_LENGTH_SAMPLES // 2)
            end_idx_truncate = start_idx_truncate + FIXED_LENGTH_SAMPLES
            
            # Ensure we don't go out of bounds if event is too close to start/end of a short slice
            if end_idx_truncate > current_len:
                end_idx_truncate = current_len
                start_idx_truncate = max(0, end_idx_truncate - FIXED_LENGTH_SAMPLES)

            processed_trace_data = sliced_data_normalized[start_idx_truncate:end_idx_truncate]
            # If still not FIXED_LENGTH_SAMPLES (e.g. original slice was too short), pad
            if len(processed_trace_data) < FIXED_LENGTH_SAMPLES:
                 processed_trace_data = np.pad(processed_trace_data, (0, FIXED_LENGTH_SAMPLES - len(processed_trace_data)), 'constant')

        elif current_len < FIXED_LENGTH_SAMPLES:
            processed_trace_data = np.pad(sliced_data_normalized, (0, FIXED_LENGTH_SAMPLES - current_len), 'constant')
        else:
            processed_trace_data = sliced_data_normalized
        
        if len(processed_trace_data) != FIXED_LENGTH_SAMPLES:
            print(f"  FATAL ERROR: Final trace length {len(processed_trace_data)} != {FIXED_LENGTH_SAMPLES} for {mseed_filepath}. This should not happen.")
            return None, None


        # 5. Calculate auxiliary data (std dev before and after event within the *original* slice)
        # The event time relative to the *start of the slice* is TIME_BEFORE_EVENT_SEC
        # (unless the slice was truncated at the beginning of the file)
        
        event_index_in_slice = int(TIME_BEFORE_EVENT_SEC * TARGET_SAMPLING_RATE_HZ)
        if slice_start_sec_rel == 0: # If the slice started at the beginning of the file
            event_index_in_slice = int(relative_event_time_sec * TARGET_SAMPLING_RATE_HZ)
        
        event_index_in_slice = min(event_index_in_slice, len(sliced_data) -1) # Cap at end of data
        event_index_in_slice = max(0, event_index_in_slice) # Ensure non-negative


        data_before_event_in_slice = sliced_data[:event_index_in_slice]
        data_after_event_in_slice = sliced_data[event_index_in_slice:]

        std_before = np.std(data_before_event_in_slice) if len(data_before_event_in_slice) > 0 else 0.0
        std_after = np.std(data_after_event_in_slice) if len(data_after_event_in_slice) > 0 else 0.0
        
        aux_data_array = np.array([std_before, std_after], dtype=np.float32)

        return processed_trace_data, aux_data_array

    except Exception as e:
        print(f"  Unhandled error processing event from {mseed_filepath} at {relative_event_time_sec}s: {e}")
        return None, None


# --- Main Script Logic ---
def main():
    # Find all subdirectories in the base training directory
    # These are expected to be 'lunar', 'mars', etc.
    source_type_folders = [f.path for f in os.scandir(BASE_TRAINING_DIR) if f.is_dir()]

    if not source_type_folders:
        print(f"No subdirectories found in {BASE_TRAINING_DIR}. Exiting.")
        return

    for source_folder_path in source_type_folders:
        source_name = os.path.basename(source_folder_path)
        print(f"\nProcessing source type: {source_name}")

        catalog_path = os.path.join(source_folder_path, CATALOG_FILENAME)
        if not os.path.exists(catalog_path):
            print(f"  Catalog file '{CATALOG_FILENAME}' not found in {source_folder_path}. Skipping this source.")
            continue

        # Create output directories if they don't exist
        output_trace_dir = os.path.join(source_folder_path, TRACE_NPA_DIR_NAME)
        output_aux_dir = os.path.join(source_folder_path, AUX_NPA_DIR_NAME)
        os.makedirs(output_trace_dir, exist_ok=True)
        os.makedirs(output_aux_dir, exist_ok=True)

        try:
            df_catalog = pd.read_csv(catalog_path)
        except Exception as e:
            print(f"  Error reading catalog {catalog_path}: {e}. Skipping this source.")
            continue

        print(f"  Found {len(df_catalog)} events in catalog for {source_name}.")
        
        successful_events = 0
        for idx, row in df_catalog.iterrows():
            # Ensure columns exist, handle potential missing columns gracefully
            try:
                # Assuming 'filename' in catalog does NOT have .mseed extension
                # And that mseed files are directly in source_folder_path
                base_mseed_filename = row['filename']
                mseed_filepath = os.path.join(source_folder_path, f"{base_mseed_filename}")
                
                # If your catalog's 'filename' column ALREADY includes '.mseed' then use:
                # mseed_filepath = os.path.join(source_folder_path, row['filename'])


                relative_event_time_sec = float(row['relative_time_sec'])
                label = int(row['label'])

            except KeyError as e:
                print(f"  Missing expected column {e} in catalog {catalog_path} at row {idx}. Skipping this event.")
                continue
            except ValueError as e:
                print(f"  Error parsing value in catalog {catalog_path} at row {idx}: {e}. Skipping this event.")
                continue


            if not os.path.exists(mseed_filepath):
                print(f"  Mseed file not found: {mseed_filepath} (referenced in catalog at row {idx}). Skipping this event.")
                continue

            # Define unique output filenames for each event's .npy files
            # Using overall_event_id if it exists, otherwise a combination
            if 'overall_event_id' in row:
                output_file_prefix = f"event_{row['overall_event_id']:06d}"
            else: # Fallback if overall_event_id is not in the catalog
                output_file_prefix = f"{os.path.splitext(base_mseed_filename)[0]}_day{row.get('day_index',0)}_evt{row.get('event_index_in_day',idx)}"

            output_trace_filepath = os.path.join(output_trace_dir, f"{output_file_prefix}_trace.npy")
            output_aux_filepath = os.path.join(output_aux_dir, f"{output_file_prefix}_aux.npy")
            output_label_filepath = os.path.join(output_aux_dir, f"{output_file_prefix}_label.npy") # Save label too

            print(f"    Processing event {idx+1}/{len(df_catalog)} from {base_mseed_filename} at {relative_event_time_sec:.2f}s...")
            
            trace_data_np, aux_data_np = process_and_save_data(
                mseed_filepath,
                relative_event_time_sec,
            )

            if trace_data_np is not None and aux_data_np is not None:
                np.save(output_trace_filepath, trace_data_np)
                np.save(output_aux_filepath, aux_data_np)
                np.save(output_label_filepath, np.array([label], dtype=np.int8)) # Save label as a separate file
                successful_events +=1
                # print(f"      Saved: {output_trace_filepath}, {output_aux_filepath}, {output_label_filepath}")
            else:
                print(f"      Failed to process event {idx+1} from {base_mseed_filename}.")
        
        print(f"  Successfully processed and saved {successful_events} events for {source_name}.")

    print("\nAll processing finished.")

if __name__ == "__main__":
    main()