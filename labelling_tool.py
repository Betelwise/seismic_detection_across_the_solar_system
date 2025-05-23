# labeling_tool.py
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import os

# Import your refactored functions
from src.config import get_event_type_from_path, get_settings
from src.seismic_processing import (
    load_mseed_file,
    preprocess_trace,
    calculate_best_freq_range_by_pwr, # Or your preferred method
    detect_triggers_sta_lta
)
from src.file_utils import get_mseed_files, save_event_catalog, load_event_catalog

class SeismicLablerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Seismic Event Labeler")

        # --- Data and State ---
        self.mseed_files = []
        self.current_mseed_idx = -1
        self.current_st = None
        self.current_trace_processed = None
        self.current_cft = None
        self.current_triggers = [] # List of (start_sample, end_sample)
        self.current_trigger_idx = -1
        self.event_type = 'moon' # Default
        self.current_settings = get_settings(self.event_type)

        self.labeled_events = [] # List of dictionaries
        self.output_catalog_path = "labeled_event_catalog.csv" # Default
        self.reference_catalog_df = pd.DataFrame() # For Apollo catalog

        # --- GUI Elements ---
        # Frame for file operations
        file_frame = tk.Frame(master)
        file_frame.pack(pady=10)

        self.btn_load_mseed_dir = tk.Button(file_frame, text="Load Mseed Directory", command=self.load_mseed_directory)
        self.btn_load_mseed_dir.pack(side=tk.LEFT, padx=5)

        self.btn_load_ref_catalog = tk.Button(file_frame, text="Load Reference Catalog (Optional)", command=self.load_reference_catalog)
        self.btn_load_ref_catalog.pack(side=tk.LEFT, padx=5)

        self.lbl_mseed_file = tk.Label(master, text="No file loaded.")
        self.lbl_mseed_file.pack()

        # Matplotlib Figure for plotting
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Frame for navigation and labeling
        nav_frame = tk.Frame(master)
        nav_frame.pack(pady=10)

        self.btn_prev_file = tk.Button(nav_frame, text="<< Prev File", command=self.prev_file, state=tk.DISABLED)
        self.btn_prev_file.pack(side=tk.LEFT, padx=5)
        self.btn_next_file = tk.Button(nav_frame, text="Next File >>", command=self.next_file, state=tk.DISABLED)
        self.btn_next_file.pack(side=tk.LEFT, padx=5)

        self.btn_prev_event = tk.Button(nav_frame, text="< Prev Event", command=self.prev_event, state=tk.DISABLED)
        self.btn_prev_event.pack(side=tk.LEFT, padx=20)
        self.btn_next_event = tk.Button(nav_frame, text="Next Event >", command=self.next_event, state=tk.DISABLED)
        self.btn_next_event.pack(side=tk.LEFT, padx=5)

        label_frame = tk.Frame(master)
        label_frame.pack(pady=5)
        self.btn_is_event = tk.Button(label_frame, text="Is Event (1)", command=lambda: self.label_event(1), state=tk.DISABLED, bg="lightgreen")
        self.btn_is_event.pack(side=tk.LEFT, padx=5)
        self.btn_not_event = tk.Button(label_frame, text="Not Event (0)", command=lambda: self.label_event(0), state=tk.DISABLED, bg="salmon")
        self.btn_not_event.pack(side=tk.LEFT, padx=5)
        self.btn_skip_event = tk.Button(label_frame, text="Skip", command=self.skip_event, state=tk.DISABLED)
        self.btn_skip_event.pack(side=tk.LEFT, padx=5)

        self.lbl_event_info = tk.Label(master, text="Current Event: N/A")
        self.lbl_event_info.pack()

        # Save catalog
        self.btn_save_catalog = tk.Button(master, text="Save Labeled Catalog", command=self.save_labeled_catalog, state=tk.DISABLED)
        self.btn_save_catalog.pack(pady=10)


    def load_mseed_directory(self):
        dir_path = filedialog.askdirectory(title="Select Directory with Mseed Files")
        if not dir_path:
            return
        self.mseed_files = get_mseed_files(os.path.join(dir_path, "*.mseed")) # Adjust pattern if needed
        if not self.mseed_files:
            messagebox.showinfo("Info", "No mseed files found in the selected directory.")
            return

        self.output_catalog_path = os.path.join(dir_path, "labeled_events_catalog.csv")
        # Try to load existing catalog from this directory
        existing_df = load_event_catalog(self.output_catalog_path)
        if not existing_df.empty:
            self.labeled_events = existing_df.to_dict('records')
            messagebox.showinfo("Info", f"Loaded {len(self.labeled_events)} previously labeled events.")


        self.current_mseed_idx = 0
        self.load_current_mseed()
        self.update_button_states()
        self.btn_save_catalog.config(state=tk.NORMAL)

    def load_reference_catalog(self):
        filepath = filedialog.askopenfilename(title="Select Reference CSV Catalog",
                                              filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if filepath:
            try:
                self.reference_catalog_df = pd.read_csv(filepath)
                messagebox.showinfo("Success", "Reference catalog loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load reference catalog: {e}")


    def update_button_states(self):
        # File navigation
        self.btn_prev_file.config(state=tk.NORMAL if self.current_mseed_idx > 0 else tk.DISABLED)
        self.btn_next_file.config(state=tk.NORMAL if self.mseed_files and self.current_mseed_idx < len(self.mseed_files) - 1 else tk.DISABLED)

        # Event navigation and labeling
        if self.current_triggers:
            self.btn_prev_event.config(state=tk.NORMAL if self.current_trigger_idx > 0 else tk.DISABLED)
            self.btn_next_event.config(state=tk.NORMAL if self.current_trigger_idx < len(self.current_triggers) - 1 else tk.DISABLED)
            self.btn_is_event.config(state=tk.NORMAL)
            self.btn_not_event.config(state=tk.NORMAL)
            self.btn_skip_event.config(state=tk.NORMAL)
        else:
            self.btn_prev_event.config(state=tk.DISABLED)
            self.btn_next_event.config(state=tk.DISABLED)
            self.btn_is_event.config(state=tk.DISABLED)
            self.btn_not_event.config(state=tk.DISABLED)
            self.btn_skip_event.config(state=tk.DISABLED)


    def load_current_mseed(self):
        if not self.mseed_files or not (0 <= self.current_mseed_idx < len(self.mseed_files)):
            return

        filepath = self.mseed_files[self.current_mseed_idx]
        self.lbl_mseed_file.config(text=f"File: {os.path.basename(filepath)} ({self.current_mseed_idx + 1}/{len(self.mseed_files)})")

        self.event_type = get_event_type_from_path(filepath)
        self.current_settings = get_settings(self.event_type)

        self.current_st = load_mseed_file(filepath)
        if not self.current_st or not self.current_st.traces:
            messagebox.showerror("Error", f"Could not load or no traces in: {filepath}")
            self.current_trace_processed = None
            self.current_cft = None
            self.current_triggers = []
            self.current_trigger_idx = -1
            self.plot_data() # Clear plot
            return

        trace = self.current_st[0].copy() # Assuming first trace

        # --- Processing ---
        # 1. Determine best frequency (optional, could be fixed or GUI selectable)
        low_f, high_f = calculate_best_freq_range_by_pwr(
            trace.copy(), # Pass a copy to avoid modifying original trace inside function
            self.current_settings['low_freq_point'],
            self.current_settings['high_freq_point'],
            self.current_settings['frequence_window']
        )
        print(f"Using frequency range: {low_f:.1f} - {high_f:.1f} Hz for {os.path.basename(filepath)}")

        # 2. Preprocess (filter, resample, normalize)
        self.current_trace_processed = preprocess_trace(
            trace.copy(), # Pass a copy
            low_f, high_f,
            self.current_settings['resample'],
            self.current_settings.get('df_resample'), # Use .get for safety
            self.current_settings.get('clipping_std_factor')
        )

        # 3. STA/LTA
        # Use absolute values for STA/LTA input
        trace_data_for_stalta = np.abs(self.current_trace_processed.data)
        self.current_cft, on_off_indices = detect_triggers_sta_lta(
            trace_data_for_stalta,
            self.current_trace_processed.stats.sampling_rate,
            self.current_settings['sta_len'] / self.current_trace_processed.stats.sampling_rate, # Convert samples to sec
            self.current_settings['lta_len'] / self.current_trace_processed.stats.sampling_rate, # Convert samples to sec
            self.current_settings['thr_on']
        )
        self.current_triggers = on_off_indices.tolist() # Convert to list of [start, end]

        # Filter triggers: e.g. by duration
        min_duration_samples = 5 * self.current_trace_processed.stats.sampling_rate # Example: 5 seconds
        self.current_triggers = [
            trig for trig in self.current_triggers
            if (trig[1] - trig[0]) > min_duration_samples
        ]


        if self.current_triggers:
            self.current_trigger_idx = 0
            # Check if this file/trigger combo has been labeled before
            self.check_if_already_labeled()
        else:
            self.current_trigger_idx = -1
            self.lbl_event_info.config(text="No triggers found in this file.")

        self.plot_data()
        self.update_event_info_label()
        self.update_button_states()


    def plot_data(self):
        self.ax[0].clear()
        self.ax[1].clear()

        if self.current_trace_processed:
            times = self.current_trace_processed.times()
            data = self.current_trace_processed.data
            self.ax[0].plot(times, data, label="Processed Waveform")
            self.ax[0].set_ylabel("Normalized Amplitude")
            self.ax[0].grid(True)

            if self.current_cft is not None:
                # Ensure CFT times align if resampling happened
                cft_times = np.linspace(0, times[-1], len(self.current_cft), endpoint=False) # Approximate
                self.ax[1].plot(cft_times, self.current_cft, label="STA/LTA CFT", color='orange')
                self.ax[1].axhline(self.current_settings['thr_on'], color='red', linestyle='--', label=f"Thr_on={self.current_settings['thr_on']}")
                # thr_off_val = np.mean(self.current_cft) * 1.0 # Or however thr_off is determined
                thr_off_val = self.current_settings['thr_on'] / 2.0
                self.ax[1].axhline(thr_off_val, color='red', linestyle=':', label=f"Thr_off (approx)")
                self.ax[1].set_ylabel("CFT Value")
                self.ax[1].grid(True)

            # Highlight current trigger window
            if self.current_triggers and self.current_trigger_idx != -1:
                start_idx, end_idx = self.current_triggers[self.current_trigger_idx]
                start_time = times[start_idx] if start_idx < len(times) else times[0]
                end_time = times[end_idx] if end_idx < len(times) else times[-1]

                self.ax[0].axvspan(start_time, end_time, color='yellow', alpha=0.3, label="Current Event Window")
                self.ax[1].axvspan(start_time, end_time, color='yellow', alpha=0.3)

            # Plot reference arrival if available
            if not self.reference_catalog_df.empty and self.current_st:
                try:
                    # Match based on filename (you might need a more robust key)
                    filename_base = os.path.basename(self.mseed_files[self.current_mseed_idx])
                    # Extract evid or unique part from filename
                    # This part is highly dependent on your filename convention and catalog structure
                    # Example: evid = filename_base.split('_')[-1].split('.')[0]
                    # For now, let's assume the catalog has a 'filename' column
                    ref_row = self.reference_catalog_df[self.reference_catalog_df['filename'].str.contains(filename_base.split('.')[0], case=False, na=False)]
                    if not ref_row.empty and 'time_rel(sec)' in ref_row.columns:
                        arrival_time_rel = ref_row['time_rel(sec)'].iloc[0]
                        self.ax[0].axvline(arrival_time_rel, color='purple', linestyle='--', label='Ref. Arrival')
                        self.ax[0].annotate(f'Ref Arr: {arrival_time_rel:.1f}s',
                                            xy=(arrival_time_rel, self.ax[0].get_ylim()[1]*0.9),
                                            color='purple')
                except Exception as e:
                    print(f"Error plotting reference arrival: {e}")


            self.ax[0].legend(loc="upper right")
            self.ax[1].legend(loc="upper right")
            self.ax[1].set_xlabel("Time (s)")
            self.fig.suptitle(f"File: {os.path.basename(self.mseed_files[self.current_mseed_idx]) if self.current_mseed_idx !=-1 else 'N/A'}", fontsize=10)

        self.canvas.draw()

    def next_file(self):
        if self.current_mseed_idx < len(self.mseed_files) - 1:
            self.current_mseed_idx += 1
            self.load_current_mseed()

    def prev_file(self):
        if self.current_mseed_idx > 0:
            self.current_mseed_idx -= 1
            self.load_current_mseed()

    def next_event(self):
        if self.current_triggers and self.current_trigger_idx < len(self.current_triggers) - 1:
            self.current_trigger_idx += 1
            self.check_if_already_labeled()
            self.plot_data()
            self.update_event_info_label()
            self.update_button_states()


    def prev_event(self):
        if self.current_triggers and self.current_trigger_idx > 0:
            self.current_trigger_idx -= 1
            self.check_if_already_labeled()
            self.plot_data()
            self.update_event_info_label()
            self.update_button_states()

    def skip_event(self):
        # Simply move to the next event without labeling
        if self.current_trigger_idx < len(self.current_triggers) - 1:
            self.next_event()
        elif self.current_mseed_idx < len(self.mseed_files) - 1:
            self.next_file()
        else:
            messagebox.showinfo("Info", "End of files/events.")


    def check_if_already_labeled(self):
        """Check if the current file/trigger has been labeled and update button appearance."""
        # Reset button backgrounds
        self.btn_is_event.config(relief=tk.RAISED, bg="lightgreen")
        self.btn_not_event.config(relief=tk.RAISED, bg="salmon")

        if not self.mseed_files or self.current_mseed_idx == -1 or \
           not self.current_triggers or self.current_trigger_idx == -1:
            return

        filepath = self.mseed_files[self.current_mseed_idx]
        filename_base = os.path.basename(filepath)
        start_idx, end_idx = self.current_triggers[self.current_trigger_idx]
        
        # Find if this event exists in self.labeled_events
        for i, event_data in enumerate(self.labeled_events):
            if event_data.get('original_filename') == filename_base and \
               event_data.get('trigger_start_sample') == start_idx and \
               event_data.get('trigger_end_sample') == end_idx:
                
                label = event_data.get('label')
                if label == 1:
                    self.btn_is_event.config(relief=tk.SUNKEN, bg="darkgreen")
                elif label == 0:
                    self.btn_not_event.config(relief=tk.SUNKEN, bg="darkred")
                return # Found
        

    def label_event(self, label_value): # label_value: 1 for event, 0 for not event
        if not self.mseed_files or self.current_mseed_idx == -1 or \
           not self.current_triggers or self.current_trigger_idx == -1:
            messagebox.showerror("Error", "No event selected to label.")
            return

        filepath = self.mseed_files[self.current_mseed_idx]
        filename_base = os.path.basename(filepath)
        trace = self.current_trace_processed
        start_idx, end_idx = self.current_triggers[self.current_trigger_idx]

        # Ensure indices are within bounds of trace data
        start_idx = max(0, start_idx)
        end_idx = min(len(trace.data) -1, end_idx)
        if start_idx >= end_idx:
            print(f"Warning: Invalid trigger window for {filename_base} - start_idx {start_idx}, end_idx {end_idx}. Skipping label.")
            self.skip_event() # or some other handling
            return


        start_time_abs = trace.stats.starttime + (start_idx / trace.stats.sampling_rate)
        end_time_abs = trace.stats.starttime + (end_idx / trace.stats.sampling_rate)
        start_time_rel = start_idx / trace.stats.sampling_rate
        end_time_rel = end_idx / trace.stats.sampling_rate

        # Extract the actual event window data (e.g., for saving later or feature extraction)
        # window_data = trace.data[start_idx : end_idx + 1] # This can be large

        event_data = {
            'original_filename': filename_base,
            'event_type': self.event_type, # moon or mars
            'processed_trace_path': None, # Placeholder: you might save snippets later
            'trigger_start_sample': start_idx,
            'trigger_end_sample': end_idx,
            'trigger_start_time_relative_sec': round(start_time_rel, 3),
            'trigger_end_time_relative_sec': round(end_time_rel, 3),
            'trigger_start_time_utc': start_time_abs.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'trigger_end_time_utc': end_time_abs.strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'filter_low_hz': round(self.current_trace_processed.stats.processing[0].split('freqmin=')[1].split(',')[0],2) if 'bandpass' in self.current_trace_processed.stats.processing[0] else None,
            'filter_high_hz': round(self.current_trace_processed.stats.processing[0].split('freqmax=')[1].split(')')[0],2) if 'bandpass' in self.current_trace_processed.stats.processing[0] else None,
            'label': label_value # 1 or 0
        }

        # Check if this exact event (file + trigger window) was already labeled and update it
        # This is important if the user re-labels an event
        found_existing = False
        for i, existing_event in enumerate(self.labeled_events):
            if existing_event['original_filename'] == event_data['original_filename'] and \
               existing_event['trigger_start_sample'] == event_data['trigger_start_sample'] and \
               existing_event['trigger_end_sample'] == event_data['trigger_end_sample']:
                self.labeled_events[i] = event_data # Update existing
                found_existing = True
                break
        if not found_existing:
            self.labeled_events.append(event_data)

        # Update button appearance immediately
        if label_value == 1:
            self.btn_is_event.config(relief=tk.SUNKEN, bg="darkgreen")
            self.btn_not_event.config(relief=tk.RAISED, bg="salmon")
        elif label_value == 0:
            self.btn_not_event.config(relief=tk.SUNKEN, bg="darkred")
            self.btn_is_event.config(relief=tk.RAISED, bg="lightgreen")


        print(f"Labeled: {filename_base}, Trigger {self.current_trigger_idx+1}, Label: {label_value}")

        # Auto-advance to next event or file
        if self.current_trigger_idx < len(self.current_triggers) - 1:
            self.next_event()
        elif self.current_mseed_idx < len(self.mseed_files) - 1:
            self.next_file()
        else:
            messagebox.showinfo("Info", "All events in all files processed/labeled.")
            self.update_button_states() # Disable labeling buttons if at end


    def update_event_info_label(self):
        if self.current_triggers and self.current_trigger_idx != -1:
            start_idx, end_idx = self.current_triggers[self.current_trigger_idx]
            info_text = (f"Current Event: {self.current_trigger_idx + 1} / {len(self.current_triggers)} "
                         f"(Samples: {start_idx}-{end_idx})")
            self.lbl_event_info.config(text=info_text)
        elif self.current_trace_processed: # File loaded but no triggers
            self.lbl_event_info.config(text="No triggers found for current settings.")
        else: # No file loaded
            self.lbl_event_info.config(text="Current Event: N/A")


    def save_labeled_catalog(self):
        if not self.labeled_events:
            messagebox.showinfo("Info", "No events have been labeled yet.")
            return

        # Define columns for the CSV file
        # Ensure all keys used in event_data are here
        columns = [
            'original_filename', 'event_type', 'processed_trace_path',
            'trigger_start_sample', 'trigger_end_sample',
            'trigger_start_time_relative_sec', 'trigger_end_time_relative_sec',
            'trigger_start_time_utc', 'trigger_end_time_utc',
            'filter_low_hz', 'filter_high_hz', 'label'
        ]
        
        # Check if output_catalog_path is set, prompt if not
        if not self.output_catalog_path or self.output_catalog_path == "labeled_event_catalog.csv": # Default name means not set by load_mseed_directory
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Labeled Catalog As",
                initialfile="labeled_events_catalog.csv"
            )
            if not save_path:
                return # User cancelled
            self.output_catalog_path = save_path

        save_event_catalog(self.labeled_events, self.output_catalog_path, columns)
        messagebox.showinfo("Success", f"Catalog saved to {self.output_catalog_path}")


if __name__ == "__main__":
    root = tk.Tk()
    gui = SeismicLablerGUI(root)
    root.mainloop()