import os
import glob
import random
import string

import numpy as np
import matplotlib.pyplot as plt
import imageio  # for video/GIF generation
from mpl_toolkits.mplot3d import Axes3D  # Import for 3d plotting


def random_string(string_length):
    # Define the character set for the random string
    chars = string.ascii_letters + string.digits

    # Generate a random string
    return ''.join(random.choice(chars) for _ in range(string_length))


# Function to plot a single frame with 3D keypoints
def plot_frame(frame_data, frame_number):

    print(f"processing frame {frame_number}")

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
    # Create the figure with the specified size
    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(111, projection="3d")  # Initialize 3D subplot

    # Clear previous plot (if applicable)
    ax.cla()

    # Plot keypoints in 3D
    ax.scatter3D(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], c="red")

    # for joint in skeleton:
    #     start_point = frame_data[joint[0]]
    #     end_point = frame_data[joint[1]]
    #     # Draw connections between points
    #     ax.plot3D(
    #         [start_point[0], end_point[0]],
    #         [start_point[1], end_point[1]],
    #         [start_point[2], end_point[2]],
    #         color="blue",
    #     )

    ax.set_title(f"Frame: {frame_number}")
    ax.set_xlim(-1.3, 1.3)  # Set limits based on data3
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)  # Set z-axis limits based on your data range

    ax.view_init(azim=90, elev=90)

    # plt.draw()
    # plt.pause(0.001)  # Short pause to avoid rapid flickering
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def visualize_keypoints3d(keypoints, name=None):
    """
    Args:
        keypoints: Numpy array with shape (num_frames, num_keypoints, 2), typically num_keypoints=17
    """

    # Create video or GIF (optional)
    frames = []

    for i in range(len(keypoints)):
        

        frame_data = plot_frame(
            keypoints[i], i + 1, width, height, depth
        )  # Plot keypoints on each frame
        frames.append(frame_data)

    if name is None:
        # generate a random name
        name = random_string(8)

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

        print(filename)

        visualize_keypoints3d(data)

        break
