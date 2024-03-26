import os
import glob
import numpy as np


if __name__ == "__main__":

    customset_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "custom_dataset"
    )

    res2d_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "detectron2d"
    )

    res3d_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "results3d"
    )

    # Load 2D pose data
    res2d_files = glob.glob(os.path.join(res2d_dir, "*.npz"))

    for res2d_file in res2d_files:
        res2d = np.load(res2d_file, allow_pickle=True)
        # print(res2d["keypoints"])

        keypoints = res2d["keypoints"][:, 1:]

        # print(res2d["keypoints"][0, 1])
        print(keypoints)

        break

    # Load 3D pose data
    res3d_files = glob.glob(os.path.join(res3d_dir, "*.npy"))

    for res3d_file in res3d_files:
        res3d = np.load(res3d_file)
        print(res3d.shape)

        break
