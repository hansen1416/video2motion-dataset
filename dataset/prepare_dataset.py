import os
import json
from dotenv import load_dotenv

from tqdm import tqdm
import numpy as np
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

"""
Memory-Mapped Files:
import mmap

class MyDataset(Dataset):
    def __init__(self, filepath):
        self.file = open(filepath, "rb")
        self.data_mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def __getitem__(self, index):
        # Calculate offset and size based on your data structure
        offset, size = ...
        data = self.data_mmap[offset:offset+size]
        # Process or return data as needed

    def __len__(self):
        # Calculate total data size or number of elements based on file size and structure

# Close the file when done
dataset = MyDataset("large_file.data")
...
dataset.file.close()
"""


"""
Streaming:
class MyDataset(Dataset):
    def __init__(self, filepath):
        self.file = open(filepath, "rb")

    def __getitem__(self, index):
        # Read data in chunks based on your needs
        data = self.file.read(chunk_size)
        # Process or return data as needed

    def __len__(self):
        # Calculate total data size or number of elements based on file size and structure

# No need to close the file explicitly since it's handled by the garbage collector
dataset = MyDataset("large_file.data")
"""

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


def generate_meidiapipe_paths(humanoid_name, mediapipe_dir, animation_names):

    # get current absolute path

    # filename = os.path.join(os.path.dirname(__file__), "mapping", "mediapipe_paths.pkl")

    # if os.path.isfile(filename):
    #     print(f"{filename} already exists")
    #     return

    data_paths = []

    humanoid_path = os.path.join(mediapipe_dir, humanoid_name)

    # for animation_name in os.listdir(humanoid_path):
    for animation_name in animation_names:

        animation_name_path = os.path.join(humanoid_path, animation_name)

        for elevation in os.listdir(animation_name_path):

            elevation_path = os.path.join(animation_name_path, elevation)

            for azimuth in os.listdir(elevation_path):

                azimuth_path = os.path.join(elevation_path, azimuth)

                for n_frame in os.listdir(azimuth_path):

                    landmark_file = os.path.join(
                        azimuth_path, n_frame, "world_landmarks.json"
                    )

                    if not os.path.isfile(landmark_file):
                        continue

                    data_paths.append((animation_name, elevation, azimuth, n_frame))

    return data_paths

    # with open(filename, "wb") as f:
    #     pickle.dump(data_paths, f)

    # print(f"Saved {filename}")

    # return filename


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


def build_dataset(bucket):

    humanoid_name = "dors.glb"

    queue_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "queues",
    )

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    mediapipe_path = f"mediapipe/{humanoid_name}/"
    anim_euler_path = f"anim-json-euler/"

    for fname in os.listdir(queue_dir):
        with open(os.path.join(queue_dir, fname), "r") as f:
            data = json.load(f)

        # for testing, only get 100
        data = data[:500]

        features = []
        targets = []

        for animation_name, elevation, azimuth, n_frame in tqdm(data):

            try:
                landmarks_obj = bucket.get_object(
                    f"{mediapipe_path}{animation_name}/{elevation}/{azimuth}/{n_frame}/world_landmarks.json"
                )
            except oss2.exceptions.NoSuchKey as e:
                print(
                    f"oss2.exceptions.NoSuchKey: {mediapipe_path}{animation_name}/{elevation}/{azimuth}/{n_frame}/world_landmarks.json"
                )
                continue

            world_landmarks = json.loads(landmarks_obj.read())

            landmarks1d = get_landmarks1d(world_landmarks)

            features.append(landmarks1d)

            # print(landmarks1d)

            anim_euler_obj = bucket.get_object(f"{anim_euler_path}{animation_name}")

            anim_euler = json.loads(anim_euler_obj.read())

            bone_rotations = extract_anim_euler_frames(anim_euler, n_frame)

            targets.append(bone_rotations)

            # print(bone_rotations)

            # break

        features = np.array(features)
        targets = np.array(targets)

        # save features and targets to npy file
        np.save(
            os.path.join(data_dir, f"inputs_{fname.replace('.json', '')}.npy"),
            features,
        )

        np.save(
            os.path.join(data_dir, f"outputs_{fname.replace('.json', '')}.npy"), targets
        )

        break


if __name__ == "__main__":

    auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

    endpoint = "oss-ap-southeast-1.aliyuncs.com"

    # oss bucket, timeout 30s
    bucket = oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)

    build_dataset(bucket)
    pass
