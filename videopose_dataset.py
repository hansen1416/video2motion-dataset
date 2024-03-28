import os
import re
import json
import glob
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt


def anim_data_loader(json_data, start_frame=0, end_frame=30):
    """
    bones in anim-euler-json:

    ['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
    'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3',
    'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3',
    'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3',
    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
    'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3',
    'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3',
    'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3',
    'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase']

    joints in videopose3d results:

    ["pelvis","left_hip","left_knee","left_foot","right_hip","right_knee","right_foot","spine","neck","nose",
        "top","right_shoulder","right_elbow","right_hand","left_shoulder","left_elbow","left_hand",]

    bones to use corresponding to videopose3d results:

    ['Hips',
    'RightUpLeg', 'RightLeg',
    'LeftUpLeg', 'LeftLeg',
    'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
    'RightShoulder', 'RightArm', 'RightForeArm',
    'LeftShoulder', 'LeftArm', 'LeftForeArm',]
    """

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

    data_bones = []

    for bone in bones_to_use:

        values = np.array(json_data[bone])

        # we must consider the time, because the time step is not always 0.166666 ms

        # print(
        #     "values.shape",
        #     values.shape,
        #     start_frame,
        #     end_frame,
        #     values[start_frame:end_frame, :].shape,
        # )

        data_bones.append(values[start_frame:end_frame, :])

    data_bones = np.array(data_bones)
    # now the bones shape is (16, 30, 3)

    # print(data_bones.shape)

    # convert it to frame wise data, shape (30, 16, 3)
    data_bones_frame = np.transpose(data_bones, (1, 0, 2))

    # print(data_bones_frame.shape)

    return data_bones_frame

    # following code is to check if the above code is correct
    # data_bones_naive = []

    # for i in range(data_bones.shape[1]):

    #     frame_data = data_bones[:, i, :]

    #     # print(frame_data.shape)

    #     data_bones_naive.append(frame_data)

    # data_bones_naive = np.array(data_bones_naive)

    # print(data_bones_naive.shape)

    # # check if data_bones_naive and data_bones_frame are the same
    # print(np.allclose(data_bones_naive, data_bones_frame))


def concatenate3d(overwrite=False):
    anim_euler_dir = os.path.join(
        os.path.expanduser("~"),
        "Documents",
        "video2motion-animplayer",
        "public",
        "anim-euler-uniform",
    )

    res3d_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "results3d"
    )

    res3ds_dir = os.path.join(
        os.path.expanduser("~"), "Documents", "video2motion", "results3d_dataset"
    )

    os.makedirs(res3ds_dir, exist_ok=True)

    res3d_data_file = os.path.join(res3ds_dir, "res3d_data.npy")
    res3d_metadata_file = os.path.join(res3ds_dir, "res3d_metadata.pkl")
    anim_euler_data_file = os.path.join(res3ds_dir, "anim_euler_data.npy")

    if (
        overwrite == False
        and os.path.exists(res3d_data_file)
        and os.path.exists(res3d_metadata_file)
        and os.path.exists(anim_euler_data_file)
    ):
        return (
            np.load(res3d_data_file),
            pickle.load(open(res3d_metadata_file, "rb")),
            np.load(anim_euler_data_file),
        )

    # Load 3D pose data
    res3d_files = glob.glob(os.path.join(res3d_dir, "*.npy"))

    res3d_data = []
    res3d_metadata = []
    anim_euler_data = []

    interval_length = 30

    limit = -1
    counter = 0

    skip_anim = ["Run To Dive-30-0.avi.npy"]

    for res3d_file in res3d_files:

        if os.path.basename(res3d_file) in skip_anim:
            continue

        res3d = np.load(res3d_file)
        # print(res3d.shape[0])

        anim_name = os.path.basename(res3d_file).replace(".npy", "").replace(".avi", "")

        print(anim_name)

        anim_name_raw = re.sub(r"-\d+-\d+", "", anim_name)

        with open(os.path.join(anim_euler_dir, f"{anim_name_raw}.json"), "r") as f:
            anim_euler = json.load(f)

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

            euler_data = anim_data_loader(
                anim_euler, start_frame=start_frame, end_frame=end_frame
            )

            assert euler_data.shape == (
                30,
                16,
                3,
            ), f"euler_data shape is not (30, 16, 3) {str(euler_data.shape)}"

            anim_euler_data.append(euler_data)
            # return

            end_frame += interval_length

        counter += 1

        if limit > 0 and counter >= limit:
            break

    # save `res3d_data` to npy file
    res3d_data = np.array(res3d_data, dtype=np.float32)

    np.save(os.path.join(res3ds_dir, "res3d_data.npy"), res3d_data)

    # save `res3d_metadata` to pickle file
    with open(os.path.join(res3ds_dir, "res3d_metadata.pkl"), "wb") as f:
        pickle.dump(res3d_metadata, f)
    # print(res3d_data[0].shape)

    anim_euler_data = np.array(anim_euler_data, dtype=np.float32)

    print(anim_euler_data.shape)

    np.save(os.path.join(res3ds_dir, "anim_euler_data.npy"), anim_euler_data)

    return res3d_data, res3d_metadata, anim_euler_data


if __name__ == "__main__":

    from visualize import visualize_keypoints3d

    res3d_data, res3d_metadata, anim_euler_data = concatenate3d(overwrite=True)

    # print(res3d_data.shape, res3d_data.dtype)
    # print(len(res3d_metadata))

    # get a random index in the range of res3d_data.shape[0]

    # for _ in range(10):
    #     idx = random.randint(0, res3d_data.shape[0])

    #     visualize_keypoints3d(
    #         keypoints=res3d_data[idx],
    #         anim_name=f"{res3d_metadata[idx][0]}-{res3d_metadata[idx][1]}-{res3d_metadata[idx][2]}",
    #     )
