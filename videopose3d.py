import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import imageio  # for video/GIF generation
from mpl_toolkits.mplot3d import Axes3D  # Import for 3d plotting


# Function to plot a single frame with 3D keypoints
def plot_frame(frame_data, frame_number, width, height, depth):

    # Define body joint connections (modify based on your keypoint definition)
    skeleton = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],  # Head
        [5, 6],
        [5, 7],
        [7, 9],
        [6, 8],
        [8, 10],  # Arms
        [5, 11],
        [6, 12],
        [11, 12],  # Body
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16],
    ]

    fig = plt.figure()  # Create a new figure for 3D plot
    ax = fig.add_subplot(111, projection="3d")  # Initialize 3D subplot

    # Clear previous plot (if applicable)
    ax.cla()

    # Plot keypoints in 3D
    ax.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], c="red")

    for joint in skeleton:
        start_point = frame_data[joint[0]]
        end_point = frame_data[joint[1]]
        # Draw connections between points
        ax.plot3D(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            [start_point[2], end_point[2]],
            color="blue",
        )

    ax.set_title(f"Frame: {frame_number}")
    ax.set_xlim(0, width)  # Set limits based on data
    ax.set_ylim(0, height)
    ax.set_zlim(0, depth)  # Set z-axis limits based on your data range

    plt.close(fig)  # Close the figure to avoid memory leak

    plt.draw()
    plt.pause(0.001)  # Short pause to avoid rapid flickering


def visualize_keypoints3d(keypoints, width, height, depth):
    """
    Args:
        keypoints: Numpy array with shape (num_frames, num_keypoints, 2), typically num_keypoints=17
    """

    # keypoints[:, :, 0] = width - keypoints[:, :, 0]
    # keypoints[:, :, 1] = height - keypoints[:, :, 1]

    # Create video or GIF (optional)
    frames = []

    for i in range(len(keypoints)):
        frame_data = keypoints[i]

        plot_frame(
            frame_data, i + 1, width, height, depth
        )  # Plot keypoints on each frame
        frames.append(frame_data)

    # Example for GIF using imageio
    imageio.mimsave("output.gif", frames, fps=25)  # Adjust fps as needed


if __name__ == "__main__":

    data_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "results3d"
    )

    filenames = glob.glob(os.path.join(data_dir, "*.npy"))

    width = 1
    height = 1
    depth = 1

    for filename in filenames:

        data = np.load(filename)

        print(data.shape)

        visualize_keypoints3d(data, width, height, depth)

        break
