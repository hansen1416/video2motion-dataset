import numpy as np

HUMANOID_BONES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
]


def get_landmarks1d(landmarks):
    """
    Convert landmarks to 1d tensor, drop visibility and presence

    Args:
        landmarks: list of dict
    Returns:
        landmarks1d: ndarray
    """
    landmarks1d = []
    # flattten landmarks
    for l in landmarks:
        landmarks1d.append(l["x"])
        landmarks1d.append(l["y"])
        landmarks1d.append(l["z"])

    # convert landmarks to tensor
    landmarks1d = np.array(landmarks1d, dtype=np.float32)

    return landmarks1d


def extract_anim_euler_frames(anim_euler_data, n_frame):
    """
    Read the bone rotation animation data at the n_frame

    Args:
        anim_euler_data: dict
        n_frame: int
    Returns:
        bone_rotations: ndarray
    """
    bone_rotations = []

    # get data from n_frame
    for bone_name in HUMANOID_BONES:

        try:
            rotation = anim_euler_data[bone_name]["values"][int(n_frame)]
        except IndexError as e:
            # print(
            #     f"IndexError: {animation_name} {bone_name} {n_frame}, real length {len(animation_data[bone_name]['values'])}"
            # )
            # raise e
            rotation = anim_euler_data[bone_name]["values"][
                len(anim_euler_data[bone_name]["values"]) - 1
            ]

        bone_rotations.append(rotation[0])
        bone_rotations.append(rotation[1])
        bone_rotations.append(rotation[2])

    # convert bone_rotations to tensor
    bone_rotations = np.array(bone_rotations, dtype=np.float32)

    return bone_rotations.reshape(-1, 3)
