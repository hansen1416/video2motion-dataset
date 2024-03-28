import os
import glob
import random
import string

import numpy as np
import matplotlib.pyplot as plt
import imageio  # for video/GIF generation

from lib3d.lib import (
    vector_apply_euler_arr,
    quaternion_from_euler_arr,
    vector_apply_quaternion_arr,
)


def random_string(string_length):
    # Define the character set for the random string
    chars = string.ascii_letters + string.digits

    # Generate a random string
    return "".join(random.choice(chars) for _ in range(string_length))


def decode(filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    print("Processing {}".format(filename))
    data = np.load(filename, encoding="latin1", allow_pickle=True)
    bb = data["boxes"]
    kp = data["keypoints"]
    metadata = data["metadata"].item()
    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(
                np.full(4, np.nan, dtype=np.float32)
            )  # 4 bounding box coordinates
            results_kp.append(
                np.full((17, 4), np.nan, dtype=np.float32)
            )  # 17 COCO keypoints
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        best_bb = bb[i][1][best_match, :4]
        best_kp = kp[i][1][best_match].T.copy()
        results_bb.append(best_bb)
        results_kp.append(best_kp)

    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)
    kp = kp[:, :, :2]  # Extract (x, y)

    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])

    print("{} total frames processed".format(len(bb)))
    print("{} frames were interpolated".format(np.sum(~mask)))
    print("----------")

    return [
        {
            "start_frame": 0,  # Inclusive
            "end_frame": len(kp),  # Exclusive
            "bounding_boxes": bb,
            "keypoints": kp,
        }
    ], metadata


# print(kp)


# Function to plot a single frame with keypoints
def plot_frame2d(frame_data, frame_number, width, height, fig):

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

    plt.cla()  # Clear previous plot
    plt.scatter(frame_data[:, 0], frame_data[:, 1], c="red")  # Plot keypoints
    for joint in skeleton:
        start_point = frame_data[joint[0]]
        end_point = frame_data[joint[1]]
        plt.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            color="blue",
        )  # Draw connections
    plt.title(f"Frame: {frame_number}")
    plt.xlim(-1.3, 1.3)  # Set limits based on data
    plt.ylim(-1, 1)

    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data


def visualize_keypoints2d(filename):
    """
    Args:
        keypoints: Numpy array with shape (num_frames, num_keypoints, 2), typically num_keypoints=17
    """

    data_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "detectron2d"
    )

    [data_obj], metadata = decode(os.path.join(data_dir, f"{filename}.npz"))

    # start_frame = data_obj["start_frame"]
    # end_frame = data_obj["end_frame"]
    # bounding_boxes = data_obj["bounding_boxes"]
    # (num_frames, num_keypoints, 2)
    keypoints = data_obj["keypoints"]

    width, height = metadata["w"], metadata["h"]

    keypoints[:, :, 0] = (width - keypoints[:, :, 0] - (width / 2)) / (width / 2)
    keypoints[:, :, 1] = (height - keypoints[:, :, 1] - (height / 2)) / (height / 2)

    # Create the figure with the specified size
    fig = plt.figure(figsize=(16 * 2, 12 * 2))

    # Create video or GIF (optional)
    frames = []

    for i in range(len(keypoints)):

        frame_data = plot_frame2d(
            keypoints[i], i, width, height, fig
        )  # Plot keypoints on each frame
        frames.append(frame_data)

    anim_name = os.path.basename(filename).replace(".avi", "")

    os.makedirs("gifs", exist_ok=True)

    # Example for GIF using imageio
    imageio.mimsave(
        os.path.join("gifs", f"{anim_name}2d.gif"), frames, fps=50
    )  # Adjust fps as needed


# Function to plot a single frame with 3D keypoints
def plot_frame3d(frame_data, frame_number, fig, ax):

    print(f"processing frame {frame_number}")

    # Define body joint connections (modify based on your keypoint definition)
    labels = [
        "pelvis",
        "left_hip",
        "left_knee",
        "left_foot",
        "right_hip",
        "right_knee",
        "right_foot",
        "spine",
        "neck",
        "nose",
        "top",
        "right_shoulder",
        "right_elbow",
        "right_hand",
        "left_shoulder",
        "left_elbow",
        "left_hand",
    ]

    skeleton = [
        ["pelvis", "left_hip"],
        ["left_hip", "left_knee"],
        ["left_knee", "left_foot"],
        ["pelvis", "right_hip"],
        ["right_hip", "right_knee"],
        ["right_knee", "right_foot"],
        ["pelvis", "spine"],
        ["spine", "neck"],
        ["neck", "nose"],
        ["nose", "top"],
        ["neck", "right_shoulder"],
        ["right_shoulder", "right_elbow"],
        ["right_elbow", "right_hand"],
        ["neck", "left_shoulder"],
        ["left_shoulder", "left_elbow"],
        ["left_elbow", "left_hand"],
    ]

    # Clear previous plot (if applicable)
    ax.cla()

    # Plot keypoints in 3D
    ax.scatter3D(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], c="red")

    for i in range(len(frame_data)):
        ax.text(
            frame_data[i, 0],
            frame_data[i, 1],
            frame_data[i, 2],
            labels[i],
            color="black",
        )

    for a, b in skeleton:
        start_point = frame_data[labels.index(a)]
        end_point = frame_data[labels.index(b)]
        # Draw connections between points
        ax.plot3D(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            [start_point[2], end_point[2]],
            color="blue",
        )

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


def visualize_keypoints3d(filename=None, keypoints=None, anim_name=None):
    """
    Args:
        keypoints: Numpy array with shape (num_frames, num_keypoints, 2), typically num_keypoints=17
    """
    assert (
        filename is not None or keypoints is not None
    ), "Either filename or keypoints must be provided"

    data_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "results3d"
    )

    if keypoints is None:
        keypoints = np.load(os.path.join(data_dir, f"{filename}.npy"))

    # Create video or GIF (optional)
    frames = []

    # Create the figure with the specified size
    fig = plt.figure(figsize=(16 * 2, 12 * 2))

    ax = fig.add_subplot(111, projection="3d")  # Initialize 3D subplot

    for i in range(len(keypoints)):

        frame_data = plot_frame3d(
            keypoints[i], i, fig, ax
        )  # Plot keypoints on each frame
        frames.append(frame_data)

    if anim_name is None:
        if filename is not None:
            anim_name = os.path.basename(filename).replace(".avi", "")
        else:
            anim_name = random_string(8)

    os.makedirs("gifs", exist_ok=True)

    # Example for GIF using imageio
    imageio.mimsave(
        os.path.join("gifs", f"{anim_name}3d.gif"), frames, fps=50
    )  # Adjust fps as needed

    print(f"Saved 3D animation as GIF {anim_name}3d.gif")


def euler2points(frame_euler):

    bones_to_use = [
        "Hips",
        "RightUpLeg",
        "RightLeg",
        "LeftUpLeg",
        "LeftLeg",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
    ]

    print(frame_euler.shape)


def visualize_euler():

    res3ds_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "results3d_dataset"
    )

    euler_data = np.load(os.path.join(res3ds_dir, "anim_euler_data.npy"))

    for i in range(len(euler_data)):
        frame_euler = euler_data[i]

        points = euler2points(frame_euler)

        break


if __name__ == "__main__":

    filename = "Walking (9)-30-0.avi"

    # visualize_keypoints2d(filename)

    # visualize_keypoints3d(filename)

    visualize_euler()
