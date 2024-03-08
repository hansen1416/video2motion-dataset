import os

import numpy as np
import matplotlib.pyplot as plt

from constants import MEDIAPIPE_JOINED_DIR, ANIM_EULER_JOINED_DIR


def plot_features():
    """
    mediapipe range from -1 to 1
    """
    # clear plot
    plt.clf()

    with open(os.path.join(MEDIAPIPE_JOINED_DIR, "joined.npy"), "rb") as f:
        features = np.load(f)

    # floatten data tp 1d array
    features = features.flatten()

    # plot data using matplotlib
    plt.hist(features, bins=100)
    # plt.show()
    # save to a image named "features.png"
    plt.savefig(os.path.join("plot", "features.png"))


def plot_targets():
    """
    targets are range from -3.1415... to 3.1415...
    """

    # clear plot
    plt.clf()

    with open(os.path.join(ANIM_EULER_JOINED_DIR, "joined.npy"), "rb") as f:
        targets = np.load(f)

    # floatten data tp 1d array
    targets = targets.flatten()

    # plot data using matplotlib
    plt.hist(targets, bins=100)
    # plt.show()
    # save to a image named "targets.png"
    plt.savefig(os.path.join("plot", "targets.png"))


if __name__ == "__main__":

    plot_features()

    plot_targets()
