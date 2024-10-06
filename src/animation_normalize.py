import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from obspy import Stream, read
import glob

# Load the .mseed file
mseed_files = sorted(glob.glob('./data/lunar/test/data/S12_GradeB/*.mseed'))
def load_mseed(file_idx):
    st = read(mseed_files[file_idx])
    return st, mseed_files[file_idx]

st, filename = load_mseed(23)
print(filename)

# Extract the trace for determining x and y limits
tr = st.traces[0]
tr_times = tr.times()
tr_data = tr.data

# Function to normalize data
def normalize(data):
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1

# Normalize the original data
normalized_data = normalize(tr_data)

# Set up the figure and axis for the plot
fig, ax = plt.subplots()
line, = ax.plot(tr_times, tr_data, lw=2, color='blue', label='Original Data')  # Original data
line_norm, = ax.plot(tr_times, normalized_data, lw=2, color='orange', label='Normalized Data')  # Normalized data
ax.set_xlim([min(tr_times), max(tr_times)])  # X-axis limits based on trace time range
ax.set_ylim([-1, 1])  # Y-axis limits for normalized data
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
title = ax.set_title('Normalization Animation')
ax.legend()

# Create the animation
def update(frame):
    # Calculate interpolation factor (0 to 1)
    alpha = frame / 100  # You can adjust the number of frames for a smoother animation

    # Interpolate between the original and normalized data
    interpolated_data = (1 - alpha) * tr_data + alpha * normalized_data
    
    # Update the line data for the current frame
    line_norm.set_ydata(interpolated_data)
    
    return line_norm,

# Create the animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Save the animation to a MP4 file using the 'ffmpeg' writer
ani.save("normalization_animation.gif", writer='pillow')

# Show the animation in a window
plt.show()
