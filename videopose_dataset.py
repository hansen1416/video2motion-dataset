import os
import numpy as np


if __name__ == "__main__":

    customset_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "custom_dataset"
    )

    # Load the dataset
    data = np.load(os.path.join(customset_dir, "20240322-2086.npz"), allow_pickle=True)

    print(type(data["positions_2d"]))
    print(type(data["metadata"]))
