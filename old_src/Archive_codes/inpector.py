import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import Button, VBox, HBox, Output, Label
from IPython.display import display
import glob

import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import os
# Step 1: Load all CSV files
#csv_files = sorted(glob.glob('./data/lunar/training/data/S12_GradeA/test1*.csv'))
csv_files = sorted(glob.glob('E:/Projects_dont_move/Projects_Pending/Seismic Detection Across the Solar System/data/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/test1/*.csv'))

current_file_idx = 0

# Step 2: Define function to load CSV file
def load_csv(file_idx):
    df = pd.read_csv(csv_files[file_idx])
    return df




import time

def plot_event(file_idx):
    df = load_csv(file_idx)
    plt.figure(figsize=(10, 6))
    plt.plot(df['time_rel(sec)'], df['velocity(m/s)'])
    plt.title(f"Seismic Event from file: {csv_files[file_idx]}")
    plt.xlabel("Time")
    plt.ylabel("Velocity")
    plt.grid(True)
    plt.show()


# Step 4: Navigation Functions
def on_next_button_click(b):
    global current_file_idx
    if current_file_idx < len(csv_files) - 1:
        current_file_idx += 1
        update_plot()

def on_prev_button_click(b):
    global current_file_idx
    if current_file_idx > 0:
        current_file_idx -= 1
        update_plot()

def update_plot():
    output.clear_output(wait=True)
    with output:
        plot_event(current_file_idx)
        file_label.value = f"Current File: {csv_files[current_file_idx]}"

# Step 5: Set up buttons and output area
next_button = Button(description="Next Event")
prev_button = Button(description="Previous Event")
next_button.on_click(on_next_button_click)
prev_button.on_click(on_prev_button_click)

file_label = Label(f"Current File: {csv_files[current_file_idx]}")
output = Output()



# Initial plot
with output:
     plot_event(current_file_idx)

# Display buttons and plot
display(VBox([HBox([prev_button, next_button]), file_label, output]))
