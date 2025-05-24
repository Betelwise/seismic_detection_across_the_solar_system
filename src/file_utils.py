# src/file_utils.py
import glob
import os
import pandas as pd

def get_mseed_files(directory_path_pattern):
    """Gets a sorted list of mseed files matching the pattern."""
    return sorted(glob.glob(directory_path_pattern))

def save_event_catalog(event_list, output_filepath, columns):
    """Saves the list of event dictionaries to a CSV file."""
    event_df = pd.DataFrame(event_list, columns=columns)
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True) # Ensure directory exists
    event_df.to_csv(output_filepath, index=False)
    print(f"Catalog saved to {output_filepath}")

def load_event_catalog(filepath):
    """Loads an event catalog from a CSV file."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame() # Return empty DataFrame if file doesn't exist