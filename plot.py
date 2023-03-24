import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the sequence of points from the folder
points_folder = 'signs_pose/come/a_010'
points_files = sorted(os.listdir(points_folder))
points = [np.load(os.path.join(points_folder, f)) for f in points_files]

# Create a figure and axis for the animation
fig, ax = plt.subplots()

# Define a function to plot the points of a single frame
def plot_frame(frame_idx):
    ax.clear()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.scatter(points[frame_idx][:, 0], points[frame_idx][:, 1])

# Create the animation by repeatedly calling the plotting function with different frame indices
animation = FuncAnimation(fig, plot_frame, frames=len(points))

# Display the animation
plt.show()