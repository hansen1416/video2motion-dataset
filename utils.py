import os

import numpy as np
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

from constants import HOME_DIR


BlazePoseKeypoints = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "LEFT_RIGHT",
    10: "RIGHT_LEFT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}

HUMANOID_BONES_ALL = [
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
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
]

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


def load_env_from_file():
    """
    Loads environment variables from a local file.

    Args:
        file_path (str): Path to the environment variable file.

    Returns:
        dict: Dictionary containing the loaded environment variables.
    """

    # .env file is at the level of this file
    file_path = os.path.join(os.path.dirname(__file__), ".env")

    if not os.path.exists(file_path):
        raise ValueError(f"Environment variable file not found: {file_path}")

    # Open the file and read its lines
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Load variables into a dictionary
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue
        key, value = line.split("=", 1)  # Split line by `=`
        # Remove leading/trailing quotes from key and value
        os.environ[key.strip()] = value.strip()


def upload_anim_data(sub_folder_name):

    load_env_from_file()

    # 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

    # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
    endpoint = "oss-ap-southeast-1.aliyuncs.com"

    # 填写Bucket名称，并设置连接超时时间为30秒。
    bucket = oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)

    pubnlic_dir = os.path.join(
        HOME_DIR, "Repos", "video2motion-screenshots" "anim-player", "public"
    )

    anim_dir = os.path.join(pubnlic_dir, sub_folder_name)

    for filename in os.listdir(anim_dir):
        object_name = f"{sub_folder_name}/{filename}"
        local_file = os.path.join(anim_dir, filename)

        # check if `object_name` already exists in the bucket
        if bucket.object_exists(object_name):
            print(f"{object_name} already exists in the bucket")
            continue

        # print(object_name, local_file)

        # 上传文件到OSS。
        # yourObjectName由包含文件后缀，不包含Bucket名称组成的Object完整路径，例如abc/efg/123.jpg。
        # yourLocalFile由本地文件路径加文件名包括后缀组成，例如/users/local/myfile.txt。
        result = bucket.put_object_from_file(object_name, local_file)

        # HTTP返回码。
        print("{} => {} http status: {}".format(local_file, object_name, result.status))

        if int(result.status) != 200:
            # output the local file path to local log
            with open(f"upload-{sub_folder_name}-error.log", "a") as f:
                f.write(f"{local_file}\n")


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

    # upload_anim_data("anim-json")
    # upload_anim_data("anim-json-mixamo")
    # upload_anim_data("anim-json-euler")

    # join_queue(MEDIAPIPE_QUEUE_DATA_DIR, MEDIAPIPE_JOINED_DIR)
    # join_queue(RESNET_QUEUE_DATA_DIR, RESNET_JOINED_DIR)
    # join_queue(ANIM_EULER_QUEUE_DATA_DIR, ANIM_EULER_JOINED_DIR)

    check_joined()
