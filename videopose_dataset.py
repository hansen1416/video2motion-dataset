import os
import glob
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

from visualize import visualize_keypoints3d


def concatenate3d():
    res3d_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "results3d"
    )

    res3ds_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "results3d_dataset"
    )

    os.makedirs(res3ds_dir, exist_ok=True)

    res3d_data_file = os.path.join(res3ds_dir, "res3d_data.npy")
    res3d_metadata_file = os.path.join(res3ds_dir, "res3d_metadata.pkl")

    if os.path.exists(res3d_data_file) and os.path.exists(res3d_metadata_file):
        return np.load(res3d_data_file), pickle.load(open(res3d_metadata_file, "rb"))

    # Load 3D pose data
    res3d_files = glob.glob(os.path.join(res3d_dir, "*.npy"))

    res3d_data = []
    res3d_metadata = []

    interval_length = 30

    for res3d_file in res3d_files:
        res3d = np.load(res3d_file)
        # print(res3d.shape[0])

        anim_name = os.path.basename(res3d_file).replace(".npy", "").replace(".avi", "")

        end_frame = interval_length
        # get 30 frame interval for each data row
        while end_frame < (res3d.shape[0] + interval_length):
            # if end_frame > res3d.shape[0], get the last 30 frames
            # otherwise get the 30 frames before end_frame, so there might be some overlap
            if end_frame > res3d.shape[0]:
                start_frame = res3d.shape[0] - interval_length
                end_frame = res3d.shape[0]
            else:
                start_frame = end_frame - interval_length

            res3d_data.append(res3d[start_frame:end_frame])
            res3d_metadata.append(tuple([anim_name, start_frame, end_frame]))

            end_frame += interval_length

    # save `res3d_data` to npy file
    res3d_data = np.array(res3d_data, dtype=np.float32)

    np.save(os.path.join(res3ds_dir, "res3d_data.npy"), res3d_data)

    # save `res3d_metadata` to pickle file
    with open(os.path.join(res3ds_dir, "res3d_metadata.pkl"), "wb") as f:
        pickle.dump(res3d_metadata, f)
    # print(res3d_data[0].shape)

    return res3d_data, res3d_metadata


if __name__ == "__main__":

    res3d_data, res3d_metadata = concatenate3d()

    # print(res3d_data.shape, res3d_data.dtype)
    # print(len(res3d_metadata))

    # get a random index in the range of res3d_data.shape[0]

    for _ in range(10):
        idx = random.randint(0, res3d_data.shape[0])

        visualize_keypoints3d(
            keypoints=res3d_data[idx],
            anim_name=f"{res3d_metadata[idx][0]}-{res3d_metadata[idx][1]}-{res3d_metadata[idx][2]}",
        )
