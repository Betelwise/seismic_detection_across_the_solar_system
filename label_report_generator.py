# label_report_generator.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from obspy import UTCDateTime
from scipy import signal # For spectrogram if you decide to include it in snippets

# --- Your existing src imports ---
from src.config import get_event_type_from_path, get_settings
from src.seismic_processing import (
    load_mseed_file,
    preprocess_trace,
    calculate_best_freq_range_by_pwr,
    detect_triggers_sta_lta
)
# No file_utils.save_event_catalog needed here, JS will handle output format

# --- Configuration ---
MSEED_DATA_PATTERN = './old_src/data/lunar/training/data/S16_GradeB/*.mseed'
OUTPUT_HTML_FILE = "seismic_labeling_report.html"
IMAGE_OUTPUT_DIR = "labeling_images" # To store generated PNGs
SNIPPET_WINDOW_SEC = 1*60*60 # Seconds before/after event for snippet plot
MAX_FILES_TO_PROCESS = None # For testing

# Ensure image output directory exists
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

def generate_snippet_plot(trace_data, times, event_time_rel, sampling_rate, output_path):
    """Generates and saves a small plot of the event snippet."""
    plt.ioff() # Turn off interactive mode for script-based plotting
    fig, ax = plt.subplots(figsize=(4, 2)) # Small figure

    plot_start_time = max(0, event_time_rel - SNIPPET_WINDOW_SEC)
    plot_end_time = min(times[-1] if len(times)>0 else 0, event_time_rel + SNIPPET_WINDOW_SEC)

    ax.plot(times, trace_data, color='black', linewidth=0.8)
    ax.axvline(event_time_rel, color='red', linestyle='-', linewidth=1)
    ax.set_xlim(plot_start_time, plot_end_time)
    ax.set_yticks([]) # Minimalist plot
    ax.set_xticks([])
    # ax.set_title(f"Event at {event_time_rel:.1f}s", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    plt.ion() # Turn interactive mode back on if needed elsewhere (usually not for pure script)

def generate_daily_context_plot(trace_data, times, all_event_times_rel, output_path):
    """Generates and saves a plot of the full day with all detections."""
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(times, trace_data, color='grey', linewidth=0.5)
    for etime in all_event_times_rel:
        ax.axvline(etime, color='blue', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_xlim(times[0], times[-1])
    ax.set_yticks([])
    ax.set_xlabel("Time (s) in file")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    plt.ion()


def main_generator():
    mseed_files = sorted(glob.glob(MSEED_DATA_PATTERN))
    if not mseed_files:
        print(f"No mseed files found: {MSEED_DATA_PATTERN}")
        return

    if MAX_FILES_TO_PROCESS is not None:
        mseed_files = mseed_files[:MAX_FILES_TO_PROCESS]

    all_detections_for_html = [] # List of dicts, one per detection

    print("Processing files and generating plots...")
    for day_idx, mseed_filepath in enumerate(mseed_files):
        print(f"  Processing file {day_idx + 1}/{len(mseed_files)}: {os.path.basename(mseed_filepath)}")
        base_filename = os.path.basename(mseed_filepath)

        # --- Standard data loading and processing (similar to your interactive_labeler_mpl) ---
        event_type = get_event_type_from_path(mseed_filepath)
        current_settings = get_settings(event_type)
        st = load_mseed_file(mseed_filepath)
        if not st or not st.traces: continue
        original_trace = st[0].copy()

        try:
            low_f, high_f = calculate_best_freq_range_by_pwr(
                original_trace.copy(), current_settings['low_freq_point'],
                current_settings['high_freq_point'], current_settings['frequence_window']
            )
        except Exception: # Fallback
            low_f = current_settings['low_freq_point']
            high_f = current_settings['low_freq_point'] + current_settings['frequence_window']

        trace_processed = preprocess_trace(
            original_trace.copy(), low_f, high_f, current_settings['resample'],
            current_settings.get('df_resample'), current_settings.get('clipping_std_factor')
        )
        sampling_rate = trace_processed.stats.sampling_rate
        if sampling_rate == 0: continue

        # Get all detections for this file
        _, on_off_indices = detect_triggers_sta_lta(
            np.abs(trace_processed.data), sampling_rate,
            current_settings['sta_len'] / sampling_rate,
            current_settings['lta_len'] / sampling_rate,
            current_settings['thr_on']
        )
        all_triggers_in_file_samples = on_off_indices.tolist()
        min_duration_samples = int(2 * sampling_rate) # Example filter
        filtered_triggers = [
            trig for trig in all_triggers_in_file_samples
            if (trig[1] - trig[0]) > min_duration_samples
        ]

        if not filtered_triggers: continue

        # --- Generate daily context plot (once per file) ---
        daily_plot_filename = f"{os.path.splitext(base_filename)[0]}_daily.png"
        daily_plot_path = os.path.join(IMAGE_OUTPUT_DIR, daily_plot_filename)
        all_event_times_for_daily_plot = [(s/sampling_rate) for s,e in filtered_triggers]
        generate_daily_context_plot(trace_processed.data, trace_processed.times(),
                                    all_event_times_for_daily_plot, daily_plot_path)


        # --- Process each detection in the file ---
        for event_idx_in_day, (start_sample, end_sample) in enumerate(filtered_triggers):
            relative_time_sec = start_sample / sampling_rate

            # Generate snippet plot for this detection
            snippet_filename = f"{os.path.splitext(base_filename)[0]}_d{day_idx}_e{event_idx_in_day}.png"
            snippet_plot_path = os.path.join(IMAGE_OUTPUT_DIR, snippet_filename)
            generate_snippet_plot(trace_processed.data, trace_processed.times(),
                                  relative_time_sec, sampling_rate, snippet_plot_path)

            all_detections_for_html.append({
                'filename': base_filename,
                'day_index': day_idx + 1,
                'event_index_in_day': event_idx_in_day + 1,
                'relative_time_sec': round(relative_time_sec, 3),
                'snippet_image_path': os.path.join(IMAGE_OUTPUT_DIR, snippet_filename), # Relative path for HTML
                'daily_context_image_path': os.path.join(IMAGE_OUTPUT_DIR, daily_plot_filename) # Same for all events in this day
            })

    # --- Generate HTML file ---
    print(f"\nGenerating HTML report: {OUTPUT_HTML_FILE}")
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Seismic Event Labeling</title>
        <style>
            body { font-family: sans-serif; margin: 20px; }
            .day-section { border: 1px solid #ccc; margin-bottom: 20px; padding: 10px; }
            .day-header { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }
            .daily-context-img { max-width: 100%; border: 1px solid lightgray; margin-bottom:10px;}
            .detections-grid { display: flex; flex-wrap: wrap; gap: 10px; }
            .detection-item { border: 1px solid #eee; padding: 5px; width: 220px; /* Adjust as needed */ }
            .detection-item img { max-width: 200px; height: auto; display: block; margin-bottom: 5px; }
            .detection-item label { display: block; margin-top: 5px; }
            .controls { margin-top: 20px; margin-bottom: 20px; padding: 10px; border: 1px solid blue; }
            textarea { width: 100%; height: 200px; margin-top:10px;}
        </style>
    </head>
    <body>
        <h1>Seismic Event Labeling Tool</h1>
        <div class="controls">
            <button onclick="generateCatalog()">Generate Catalog CSV</button>
            <p>Instructions: Check the box for events you consider positive (earthquakes). All unchecked items will be labeled as 'not event' (0).</p>
            <textarea id="csvOutput" placeholder="CSV output will appear here..."></textarea>
        </div>
    """

    current_day_processed = -1
    for det in all_detections_for_html:
        if det['day_index'] != current_day_processed:
            if current_day_processed != -1:
                html_content += "</div></div>\n" # Close previous detections-grid and day-section
            html_content += f"<div class='day-section'>\n"
            html_content += f"  <div class='day-header'>File: {det['filename']} (Day {det['day_index']})</div>\n"
            html_content += f"  <img src='{det['daily_context_image_path']}' alt='Daily context for {det['filename']}' class='daily-context-img'>\n"
            html_content += f"  <div class='detections-grid'>\n"
            current_day_processed = det['day_index']

        html_content += f"""
        <div class="detection-item"
             data-filename="{det['filename']}"
             data-day-index="{det['day_index']}"
             data-event-index-in-day="{det['event_index_in_day']}"
             data-relative-time-sec="{det['relative_time_sec']}">
            <img src="{det['snippet_image_path']}" alt="Snippet for event at {det['relative_time_sec']:.2f}s">
            <span>Time: {det['relative_time_sec']:.2f}s (Evt {det['event_index_in_day']})</span>
            <label><input type="checkbox" class="event-checkbox"> Mark as Event</label>
        </div>
        """
    if all_detections_for_html: # Close the last day's grid and section
        html_content += "</div></div>\n"


    html_content += """
        <script>
            function generateCatalog() {
                const items = document.querySelectorAll('.detection-item');
                let csvData = "filename,day_index,event_index_in_day,relative_time_sec,label\\n"; // CSV Header
                items.forEach(item => {
                    const filename = item.dataset.filename;
                    const dayIndex = item.dataset.dayIndex;
                    const eventIndexInDay = item.dataset.eventIndexInDay;
                    const relativeTimeSec = item.dataset.relativeTimeSec;
                    const checkbox = item.querySelector('.event-checkbox');
                    const label = checkbox.checked ? 1 : 0;
                    csvData += `${filename},${dayIndex},${eventIndexInDay},${relativeTimeSec},${label}\\n`;
                });
                document.getElementById('csvOutput').value = csvData;
                // Optional: Trigger download
                // const blob = new Blob([csvData], { type: 'text/csv;charset=utf-g;' });
                // const link = document.createElement("a");
                // if (link.download !== undefined) { // feature detection
                //     const url = URL.createObjectURL(blob);
                //     link.setAttribute("href", url);
                //     link.setAttribute("download", "labeled_seismic_catalog.csv");
                //     link.style.visibility = 'hidden';
                //     document.body.appendChild(link);
                //     link.click();
                //     document.body.removeChild(link);
                // }
            }
        </script>
    </body>
    </html>
    """

    with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML report generated. Open '{OUTPUT_HTML_FILE}' in your browser to label.")
    print(f"Images saved in '{IMAGE_OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main_generator()