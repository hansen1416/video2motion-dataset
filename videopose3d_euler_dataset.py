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

    # for _, v in anim_euler_json_data.items():
    for bone in [
        "Hips",
        "Spine2",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "LeftUpLeg",
        "LeftLeg",
        "RightUpLeg",
        "RightLeg",
    ]:
        v = anim_euler_json_data[bone]
        data.append(np.array(v))
    # `data` is bone wise data, shape (num_bones, num_frames, 3)
    data = np.array(data)
    # convert it to frame wise data, shape (num_frames, num_bones, 3)
    return np.transpose(data, (1, 0, 2))

    # print(data.shape)


def concatenate3d_anim_euler():
    """
    `videopose3d_euler_dataset` contains the 3D pose data from videopose3d and euler angles from `anim-euler-uniform`
    `anim-euler-uniform` is the true animation data from mixamo, and tracks length is equal for each bone.
    """

    videopose3d_results_dir = os.path.join(BASE_DIR, "video2motion", "results3d")

    anim_eulers_dir = os.path.join(BASE_DIR, "video2motion", "anim-euler-uniform")

    # videopose3d_results = glob.iglob(os.path.join(videopose3d_results_dir, "*.npy"))

    anim_eulers = glob.iglob(os.path.join(anim_eulers_dir, "*.json"))

    # print(videopose3d_results)

    features = []
    targets = []
    metadata = []

    for f in tqdm(anim_eulers):

        anim_name = os.path.basename(f).split(".")[0]

        videopose3d_result_path = os.path.join(
            videopose3d_results_dir, f"{anim_name}-30-0.avi.npy"
        )

        if not os.path.exists(videopose3d_result_path):
            print(f"skipping {anim_name}, videopose3d no results")
            continue

        with open(f, "r") as f:
            data = json.load(f)
            data = anim_euler_frame_wise(data)

            targets.append(data.tolist())

        videopose3d_result = np.load(videopose3d_result_path)

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


def load_full_data():
    with open(
        os.path.join(
            BASE_DIR, "video2motion", "videopose3d_euler_dataset", "features.pkl"
        ),
        "rb",
    ) as f:
        features = pickle.load(f)

    with open(
        os.path.join(
            BASE_DIR, "video2motion", "videopose3d_euler_dataset", "targets.pkl"
        ),
        "rb",
    ) as f:
        targets = pickle.load(f)
    #
    with open(
        os.path.join(
            BASE_DIR, "video2motion", "videopose3d_euler_dataset", "metadata.pkl"
        ),
        "rb",
    ) as f:
        metadata = pickle.load(f)

    return features, targets, metadata


def trunk_data(max_frame=30):
    """
    get the results from `videopose3d_euler_dataset` and trunk the data to `max_frame` frames

    """

    features, targets, metadata = load_full_data()

    new_features = []
    new_targets = []
    new_metadata = []

    for i in tqdm(range(len(features))):

        total_frame = metadata[i]["total_frame"]

        if total_frame < max_frame:
            print(f"skipping {metadata[i]['name']}, total_frame < max_frame")
            continue

        for j in range(0, total_frame, max_frame):
            if j + max_frame < total_frame:
                feat = features[i][j : j + max_frame]
                targ = targets[i][j : j + max_frame]
                meta = {
                    "name": metadata[i]["name"],
                    "total_frame": max_frame,
                    "start_frame": j,
                    "end_frame": j + max_frame,
                }

            else:
                # if the remaining frames are less than max_frame
                # take the last `max_frame` frames
                feat = features[i][total_frame - max_frame :]
                targ = targets[i][total_frame - max_frame :]
                meta = {
                    "name": metadata[i]["name"],
                    "total_frame": max_frame,
                    "start_frame": total_frame - max_frame,
                    "end_frame": total_frame,
                }

            feat = np.array(feat)
            targ = np.array(targ)

            try:

                assert feat.shape == (
                    max_frame,
                    17,
                    3,
                ), f"feat shape not correct {feat.shape}, {metadata[i]['name']}"
                assert targ.shape == (
                    max_frame,
                    12,
                    3,
                ), f"targ shape not correct {targ.shape}, {metadata[i]['name']}"

            except AssertionError as e:
                print(e)
                continue

            new_features.append(feat)
            new_targets.append(targ)
            new_metadata.append(meta)

        # break

    # print(new_features)

    new_features = np.array(new_features, dtype=np.float32)
    new_targets = np.array(new_targets, dtype=np.float32)
    # new_metadata = np.array(new_metadata)

    # (17028, 30, 17, 3) (17028, 30, 12, 3) 17028
    print(new_features.shape, new_targets.shape, len(new_metadata))

    # save the new_features, new_targets to npy files
    data_dir = os.path.join(
        BASE_DIR, "video2motion", "videopose3d_euler_dataset_trunk30"
    )

    os.makedirs(data_dir, exist_ok=True)

    np.save(
        os.path.join(data_dir, "features.npy"),
        new_features,
    )

    np.save(
        os.path.join(data_dir, "targets.npy"),
        new_targets,
    )

    with open(os.path.join(data_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(new_metadata, f)

    return new_features, new_targets, new_metadata


if __name__ == "__main__":

    # concatenate3d_anim_euler()

    # features, targets, metadata = load_full_data()

    # print(len(features), len(targets), len(metadata))

    # for i in range(10):
    #     print(len(features[i]), type(features[i]))
    #     print(len(targets[i]), type(targets[i]))
    #     print(metadata[i])

    trunk_data()
