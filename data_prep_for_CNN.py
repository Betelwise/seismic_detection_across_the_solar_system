import pandas as pd
import numpy as np
from obspy import read, UTCDateTime # UTCDateTime needed for slicing
# from obspy.signal.filter import bandpass # Not directly used if st.filter is preferred
import os
import glob

# --- Configuration ---
BASE_TRAINING_DIR = './data/training' # Root directory containing 'lunar', 'mars', etc.
CATALOG_FILENAME = 'catalog.csv'      # Name of the catalog file inside each subfolder
fixed_length = 5565

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
        # st = load_mseed_file(mseed_filepath)
        st = read(mseed_filepath)
        if not st:
            print(f"  Warning: Could not read or empty stream: {mseed_filepath}")
            return None, None # Indicate failure to process this event

        event_type = get_event_type_from_path(mseed_filepath)
        if event_type is None:
            print(f"  Warning: Could not determine event type from path: {mseed_filepath}. Skipping.")
            return None, None
        current_settings = get_settings(event_type)
        original_trace = st[0].copy()

        # 1. Preprocess the trace: bandpass filter, resample, clip, normalize
        tr_filtered = preprocess_trace(
            original_trace.copy(), 
            current_settings.get('low_freq_point'), 
            current_settings.get('high_freq_point'), 
            current_settings.get('frequence_window'),
            current_settings.get('df_resample'), 
            current_settings.get('clipping_std_factor')
        )

        # 2. Define the slice window based on relative_event_time_sec
        # The times are relative to the START of the (potentially resampled) trace
        slice_start_sec_rel = relative_event_time_sec - TIME_BEFORE_EVENT_SEC
        slice_end_sec_rel = relative_event_time_sec + TIME_AFTER_EVENT_SEC
        # Cut the data from start_time to end_time
        sliced_st = tr_filtered.slice(starttime=st[0].stats.starttime + slice_start_sec_rel, 
                                    endtime=st[0].stats.starttime + slice_end_sec_rel)
        # Normalize the sliced data (mean=0, std=1)
        sliced_data = np.atleast_1d(sliced_st[0].data)
        sliced_data_normalized = (sliced_data - np.mean(sliced_data)) / np.std(sliced_data)

        # Calculate standard deviation before and after time_rel
        before_data = tr_filtered.slice(starttime=st[0].stats.starttime + slice_start_sec_rel, 
                                        endtime=st[0].stats.starttime + relative_event_time_sec)
        after_data = tr_filtered.slice(starttime=st[0].stats.starttime + relative_event_time_sec, 
                                        endtime=st[0].stats.starttime + slice_end_sec_rel)
        
        std_before = np.std(before_data[0].data)
        std_after = np.std(after_data[0].data)

        # Pad or truncate the data to the fixed length
        if len(sliced_data_normalized) > fixed_length:
            sliced_data_normalized = sliced_data_normalized[:fixed_length]
        else:
            sliced_data_normalized = np.pad(sliced_data_normalized, (0, fixed_length - len(sliced_data_normalized)), 'constant')


        
        aux_data_array = np.array([std_before, std_after], dtype=np.float32)

        return sliced_data_normalized, aux_data_array

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
        # Create empty lists to store the data and labels
        data_list = []
        labels = []
        aux_data_list = []  # To store auxiliary data (std dev before and after)
        source_name = os.path.basename(source_folder_path)
        print(f"\nProcessing source type: {source_name}")

        catalog_path = os.path.join(source_folder_path, CATALOG_FILENAME)
        if not os.path.exists(catalog_path):
            print(f"  Catalog file '{CATALOG_FILENAME}' not found in {source_folder_path}. Skipping this source.")
            continue

        # Create output directories if they don't exist
        output_np_dir = os.path.join(source_folder_path, "np_arrays")
        os.makedirs(output_np_dir, exist_ok=True)
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
                base_mseed_filename = row['filename']
                mseed_filepath = os.path.join(source_folder_path, f"{base_mseed_filename}")
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

            # output_trace_filepath = os.path.join(output_trace_dir, f"{output_file_prefix}_trace.npy")
            # output_aux_filepath = os.path.join(output_aux_dir, f"{output_file_prefix}_aux.npy")
            # output_label_filepath = os.path.join(output_aux_dir, f"{output_file_prefix}_label.npy") # Save label too

            print(f"    Processing event {idx+1}/{len(df_catalog)} from {base_mseed_filename} at {relative_event_time_sec:.2f}s...")
            
            trace_data_np, aux_data_np = process_and_save_data(
                mseed_filepath,
                relative_event_time_sec,
            )

            if trace_data_np is not None and aux_data_np is not None:
                data_list.append(trace_data_np)
                aux_data_list.append(aux_data_np)
                labels.append(label)
                # np.save(output_trace_filepath, trace_data_np)
                # np.save(output_aux_filepath, aux_data_np)
                # np.save(output_label_filepath, np.array([label], dtype=np.int8)) # Save label as a separate file
                successful_events +=1
                # print(f"      Saved: {output_trace_filepath}, {output_aux_filepath}, {output_label_filepath}")
            else:
                print(f"      Failed to process event {idx+1} from {base_mseed_filename}.")
        
        # Save all processed data for this source type
        if data_list:
            data_array = np.array(data_list, dtype=np.float32)
            aux_data_array = np.array(aux_data_list, dtype=np.float32)
            labels_array = np.array(labels, dtype=np.int8)

            # Save the processed data
            output_trace_filepath = os.path.join(output_np_dir, f"{source_name}_data.npy")
            output_aux_filepath = os.path.join(output_np_dir, f"{source_name}_aux.npy")
            output_labels_filepath = os.path.join(output_np_dir, f"{source_name}_labels.npy")

            np.save(output_trace_filepath, data_array)
            np.save(output_aux_filepath, aux_data_array)
            np.save(output_labels_filepath, labels_array)

            print(f"  Successfully saved {len(data_list)} events for {source_name}.")
        else:
            print(f"  No valid events processed for {source_name}.")
        

    print("\nAll processing finished.")

if __name__ == "__main__":
    main()