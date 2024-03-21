import os

import numpy as np
import matplotlib.pyplot as plt
import imageio  # for video/GIF generation


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


data_dir = os.path.join(
    os.path.expanduser("~"), "Documents", "video2motion", "detectron2d"
)

# filenames = os.listdir(data_dir)


# print(filenames)

# for filename in filenames:

#     data = np.load(
#         os.path.join(data_dir, filename), encoding="latin1", allow_pickle=True
#     )

#     print(data)

#     break


filename = "Zombie Turn-30-0.avi.npz"

[data_obj], metadata = decode(os.path.join(data_dir, filename))

start_frame = data_obj["start_frame"]
end_frame = data_obj["end_frame"]
bounding_boxes = data_obj["bounding_boxes"]
keypoints = data_obj["keypoints"]

print(start_frame, end_frame)
print(bounding_boxes.shape)
print(keypoints.shape)
print(metadata)

"""

keypoints = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
"""

print(keypoints[0])

# print(kp)


# Function to plot a single frame with keypoints
def plot_frame(frame_data, frame_number, width, height):

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
    plt.xlim(0, width)  # Set limits based on data
    plt.ylim(0, height)
    # plt.axis("off")  # Hide axis for clean visuals
    plt.draw()
    plt.pause(0.001)  # Short pause to avoid rapid flickering


def visualize_keypoints2d(keypoints, width, height):
    """
    Args:
        keypoints: Numpy array with shape (num_frames, num_keypoints, 2), typically num_keypoints=17
    """

    keypoints[:, :, 0] = width - keypoints[:, :, 0]
    keypoints[:, :, 1] = height - keypoints[:, :, 1]

    # Create video or GIF (optional)
    frames = []

    for i in range(len(keypoints)):
        frame_data = keypoints[i]

        plot_frame(frame_data, i + 1, width, height)  # Plot keypoints on each frame
        frames.append(frame_data)

    # Choose video or GIF creation method based on your preference (replace with appropriate library calls)
    # Example for video using OpenCV (install OpenCV if needed)
    # import cv2
    # video_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 25.0, (frame.shape[1], frame.shape[0]))
    # for frame in frames:
    #   video_writer.write(frame)
    # video_writer.release()

    # Example for GIF using imageio
    imageio.mimsave("output.gif", frames, fps=25)  # Adjust fps as needed


visualize_keypoints2d(keypoints, metadata["w"], metadata["h"])
