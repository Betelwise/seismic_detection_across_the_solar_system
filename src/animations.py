# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from obspy import Stream, read
# import glob

# # Load the .mseed file
# mseed_files = sorted(glob.glob('./src/data/mars/test/data/*.mseed'))
# def load_mseed(file_idx):
#     st = read(mseed_files[file_idx])
#     return st, mseed_files[file_idx]

# st, filename = load_mseed(0)
# print(filename)

# # Generate a list of frequency ranges for the animation
# # Start from 0.1 Hz to 1 Hz in increments of 0.25 Hz
# freq_ranges = [(0.1 + i * 0.025, 0.1 + i * 0.025 + 1.0) for i in range(36)]  # Up to 1 Hz
# duration = 10  # Total animation duration in seconds (increased for longer playback)
# interval = duration * 1000 // len(freq_ranges)  # Time for each frame in ms

# # Extract the trace for determining x and y limits
# tr = st.traces[0]
# tr_times = tr.times()
# tr_data = tr.data

# # Set up the figure and axis for the plot
# fig, ax = plt.subplots()
# line, = ax.plot([], [], lw=2)  # Empty line object to update

# # Set dynamic x and y limits
# ax.set_xlim([min(tr_times), max(tr_times)])  # X-axis limits based on trace time range
# ax.set_ylim([-1, 1])  # Y-axis limits (since we normalize data between -1 and 1)

# ax.set_ylabel('Velocity (m/s)')
# ax.set_xlabel('Time (s)')
# title = ax.set_title('')

# # This function will update the plot for each frame in the animation
# def update(frame):
#     # Copy the original trace for filtering
#     st_filt = st.copy()
    
#     # Get the current frequency range
#     f_min, f_max = freq_ranges[frame]

#     # Apply bandpass filter for the current frequency range
#     st_filt.filter('bandpass', freqmin=f_min, freqmax=f_max)
    
#     # Extract filtered trace data
#     tr_filt = st_filt.traces[0].copy()
#     tr_times_filt = tr_filt.times()
#     tr_data_filt = tr_filt.data

#     # Normalize the filtered data between -1 and 1
#     tr_data_filt = 2 * (tr_data_filt - np.min(tr_data_filt)) / (np.max(tr_data_filt) - np.min(tr_data_filt)) - 1

#     # Update the line data for the current frame
#     line.set_data(tr_times_filt, tr_data_filt)

#     # Update the title with the current frequency range
#     title.set_text(f'Filtered Time Series: {f_min:.1f}-{f_max:.1f} Hz')

#     return line, title

# # Create the animation
# ani = FuncAnimation(fig, update, frames=len(freq_ranges), interval=interval, blit=True)

# # Save the animation to a MP4 file using the 'ffmpeg' writer
# ani.save("src/frequency_animation.gif", writer='pillow')


# # Show the animation in a window
# plt.show()










import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from obspy import Stream, read
import glob

# Load the .mseed file
mseed_files = sorted(glob.glob('./src/data/lunar/test/data/S15_GradeA/*.mseed'))
def load_mseed(file_idx):
    st = read(mseed_files[file_idx])
    return st, mseed_files[file_idx]

st, filename = load_mseed(0)
print(filename)

# Extract the trace for determining x and y limits
tr = st.traces[0]
tr_times = tr.times()
tr_data = tr.data

# Set up the figure and axis for the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)  # Empty line object to update

# Set dynamic x and y limits
ax.set_xlim([min(tr_times), max(tr_times)])  # X-axis limits based on trace time range
ax.set_ylim([np.min(tr_data), np.max(tr_data)])  # Initial Y-axis limits for raw data

ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
title = ax.set_title('')

# Number of steps for gradual normalization
num_steps = 50
duration = 10  # Total animation duration in seconds
interval = duration * 1000 // num_steps  # Time for each frame in ms

# Compute fully normalized data between -1 and 1
tr_data_normalized = 2 * (tr_data - np.min(tr_data)) / (np.max(tr_data) - np.min(tr_data)) - 1

# This function will update the plot for each frame in the animation
def update_normalization(frame):
    # Gradually normalize the data over frames
    factor = frame / num_steps  # Normalization factor from 0 to 1
    tr_data_partial_norm = tr_data * (1 - factor) + tr_data_normalized * factor
    
    # Update the line data for the current frame
    line.set_data(tr_times, tr_data_partial_norm)

    # Update the title to reflect normalization progress
    title.set_text(f'Normalization Progress: {factor * 100:.1f}%')

    # Update Y-axis dynamically to reflect changing data range
    ax.set_ylim([np.min(tr_data_partial_norm), np.max(tr_data_partial_norm)])

    return line, title

# Create the animation for normalization
ani = FuncAnimation(fig, update_normalization, frames=num_steps, interval=interval, blit=True)

# Save the animation to a file
ani.save("src/normalization_animation.gif", writer='pillow')

# Show the animation in a window
plt.show()

