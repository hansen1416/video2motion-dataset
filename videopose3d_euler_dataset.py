import os
import glob
import json
import pickle
from tqdm import tqdm

import numpy as np

from constants import BASE_DIR


def anim_euler_frame_wise(anim_euler_json_data):
    """
    Bones in the order of:
    ['Hips', 'Spine2', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'RightShoulder', 'RightArm', 'RightForeArm',
    'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg'])

    """
    data = []

    for _, v in anim_euler_json_data.items():
        data.append(np.array(v))
    # `data` is bone wise data, shape (num_bones, num_frames, 3)
    data = np.array(data)
    # convert it to frame wise data, shape (num_frames, num_bones, 3)
    return np.transpose(data, (1, 0, 2))

    # print(data.shape)


def concatenate3d_anim_euler():

    videopose3d_results_dir = os.path.join(BASE_DIR, "video2motion", "results3d")

    anim_eulers_dir = os.path.join(BASE_DIR, "video2motion", "anim-calculated-euler")

    # videopose3d_results = glob.iglob(os.path.join(videopose3d_results_dir, "*.npy"))

    anim_eulers = glob.iglob(os.path.join(anim_eulers_dir, "*.json"))

    # print(videopose3d_results)

    features = []
    targets = []
    metadata = []

    for f in tqdm(anim_eulers):

        anim_name = os.path.basename(f).split(".")[0]

        videopose3d_result_path = os.path.join(
            videopose3d_results_dir, f"{anim_name}.avi.npy"
        )

        if not os.path.exists(videopose3d_result_path):
            print(f"skipping {anim_name}, videopose3d no results")
            continue

        with open(f, "r") as f:
            data = json.load(f)
            data = anim_euler_frame_wise(data)

            targets.append(data.tolist())

        videopose3d_result = np.load(
            os.path.join(videopose3d_results_dir, f"{anim_name}.avi.npy")
        )

        features.append(videopose3d_result.tolist())

        metadata.append(
            {
                "name": anim_name,
                "total_frame": videopose3d_result.shape[0],
            }
        )

        # break

    print(len(features), len(targets), len(metadata))

    with open(
        os.path.join(
            BASE_DIR, "video2motion", "videopose3d_euler_dataset", "features.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(features, f)

    with open(
        os.path.join(
            BASE_DIR, "video2motion", "videopose3d_euler_dataset", "targets.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(targets, f)

    with open(
        os.path.join(
            BASE_DIR, "video2motion", "videopose3d_euler_dataset", "metadata.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(metadata, f)


if __name__ == "__main__":
    concatenate3d_anim_euler()
