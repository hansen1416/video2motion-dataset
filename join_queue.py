import os

import numpy as np

from constants import (
    MEDIAPIPE_QUEUE_DATA_DIR,
    RESNET_QUEUE_DATA_DIR,
    ANIM_EULER_QUEUE_DATA_DIR,
    MEDIAPIPE_JOINED_DIR,
    RESNET_JOINED_DIR,
    ANIM_EULER_JOINED_DIR,
)


def join_queue(queue_data_dir, joined_dir, overwrite=False):

    target_file = os.path.join(joined_dir, "joined.npy")

    if overwrite == False and os.path.exists(target_file):
        print("joined data already exists at {}".format(target_file))
        return

    data = []

    for q in os.listdir(queue_data_dir):

        q_data = np.load(os.path.join(queue_data_dir, q))

        print("queue data shape", q_data.shape)

        data.append(q_data)

    data = np.concatenate(data, axis=0)

    print("joined data shape", data.shape)

    if not os.path.exists(joined_dir):
        os.makedirs(joined_dir)

    np.save(target_file, data)


def check_joined():

    mediapipe = np.load(os.path.join(MEDIAPIPE_JOINED_DIR, "joined.npy"))

    print("mediapipe joined data shape", mediapipe.shape)

    resnet = np.load(os.path.join(RESNET_JOINED_DIR, "joined.npy"))

    print("resnet joined data shape", resnet.shape)

    anim_euler = np.load(os.path.join(ANIM_EULER_JOINED_DIR, "joined.npy"))

    print("anim euler joined data shape", anim_euler.shape)


if __name__ == "__main__":

    join_queue(MEDIAPIPE_QUEUE_DATA_DIR, MEDIAPIPE_JOINED_DIR)

    join_queue(RESNET_QUEUE_DATA_DIR, RESNET_JOINED_DIR)

    join_queue(ANIM_EULER_QUEUE_DATA_DIR, ANIM_EULER_JOINED_DIR)

    check_joined()
