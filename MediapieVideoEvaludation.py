import os
import shutil
import random
import string
import json
from typing import List
from multiprocessing import Process

import cv2
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import mediapipe as mp
from dotenv import load_dotenv
import numpy as np

load_dotenv()


def generate_random_string(length=10):
    """Generates a random string of the specified length.

    Args:
        length (int, optional): The desired length of the random string. Defaults to 10.

    Returns:
        str: A random string containing letters and digits.
    """

    letters_and_digits = string.ascii_letters + string.digits
    result = "".join(random.choice(letters_and_digits) for i in range(length))
    return result


class OSSObjectTmpReader:
    def __init__(self, object_name, bucket):
        self.object_name = object_name
        self.bucket = bucket

    def __enter__(self):
        object_stream = self.bucket.get_object(self.object_name)

        extension = self.object_name.split(".")[-1]

        tmp_video_dir = os.path.join(".", "tmp")

        # create the directory if it does not exist
        os.makedirs(tmp_video_dir, exist_ok=True)

        # generate a random string for filename
        video_filename = generate_random_string() + "." + extension

        self.tmp_video_path = os.path.join(tmp_video_dir, video_filename)

        with open(self.tmp_video_path, "wb") as local_fileobj:
            shutil.copyfileobj(object_stream, local_fileobj)

        return self.tmp_video_path

    def __exit__(self, *args):
        # remove the temporary video file `self.tmp_video_path`
        os.remove(self.tmp_video_path)


class MediapipeVideoEulerData(Process):

    def __init__(
        self, bucket, process_number, anim_euler_object_keys: List, num_frames=30
    ) -> None:

        Process.__init__(self)

        self.bucket = bucket
        self.process_number = process_number
        self.anim_euler_object_keys = anim_euler_object_keys
        self.num_frames = num_frames

        self.path_split = "/" if self.bucket else os.path.sep

        model_path = os.path.join("models", "mediapipe", "pose_landmarker_lite.task")

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a pose landmarker instance with the video mode:
        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
        )

        self.PoseLandmarker = mp.tasks.vision.PoseLandmarker

    def check_frames(self, limit=10):

        counter = 0

        # 列举fun文件夹下的所有文件，包括子目录下的文件。
        for obj in oss2.ObjectIterator(self.bucket, prefix="anim-euler-uniform/"):
            animation_name = obj.key.split(self.path_split)[-1].split(".")[0]

            with OSSObjectTmpReader(obj.key, self.bucket) as tmp_file:
                with open(tmp_file, "r") as f:
                    json_data = json.load(f)

                    frame_lenth = len(json_data["Hips"])

            with OSSObjectTmpReader(
                f"videos/{animation_name}-30-0.avi", self.bucket
            ) as tmp_video_file:

                # Use OpenCV’s VideoCapture to load the input video.
                video_capture = cv2.VideoCapture(tmp_video_file)

                num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

                video_capture.release()

            assert (
                frame_lenth == num_frames
            ), f"Frame length does not match the video length. {frame_lenth} != {num_frames}, {animation_name}"

            counter += 1

            if counter > limit:
                break

    @staticmethod
    def _anim_data2frame_wise(json_data) -> np.ndarray:
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

        joints in mediapipe results:

        0 - nose
        1 - left eye (inner)
        2 - left eye
        3 - left eye (outer)
        4 - right eye (inner)
        5 - right eye
        6 - right eye (outer)
        7 - left ear
        8 - right ear
        9 - mouth (left)
        10 - mouth (right)
        11 - left shoulder
        12 - right shoulder
        13 - left elbow
        14 - right elbow
        15 - left wrist
        16 - right wrist
        17 - left pinky
        18 - right pinky
        19 - left index
        20 - right index
        21 - left thumb
        22 - right thumb
        23 - left hip
        24 - right hip
        25 - left knee
        26 - right knee
        27 - left ankle
        28 - right ankle
        29 - left heel
        30 - right heel
        31 - left foot index
        32 - right foot index

        bones to use as prediction target:

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

            data_bones.append(values)

        # now the bones shape is (num_bones, frame_len, 3)
        data_bones = np.array(data_bones)

        # print(data_bones.shape)

        # convert it to frame wise data, shape (frame_len, num_bones, 3)
        data_bones = np.transpose(data_bones, (1, 0, 2))

        # print(data_bones.shape)

        return data_bones

    def _read_animation_frames(self, animation_object) -> np.ndarray:
        """
        read the animation frames from the json file

        Args:
            animation_object (str): the object name in the oss bucket

        Returns:
            np.ndarray: the animation frames, shape (frame_len, num_bones, 3)
        """

        if self.bucket:
            with OSSObjectTmpReader(animation_object, self.bucket) as tmp_file:
                with open(tmp_file, "r") as f:
                    json_data = json.load(f)

            return self._anim_data2frame_wise(json_data)
        else:
            with open(animation_object, "r") as f:
                json_data = json.load(f)

            return self._anim_data2frame_wise(json_data)

    def _read_video_frames(self, video_filename) -> np.ndarray:
        """
        read the video frames from the video file using opencv

        Args:
            video_filename (str): the video filename in the oss bucket

        Returns:
            np.ndarray: the video frames, shape (frame_len, height, width, 3)
        """

        video_frames = []

        if self.bucket:
            with OSSObjectTmpReader(video_filename, self.bucket) as tmp_video_file:

                # Use OpenCV’s VideoCapture to load the input video.
                video_capture = cv2.VideoCapture(tmp_video_file)

                while video_capture.isOpened():

                    ret, frame = video_capture.read()

                    if not ret:
                        break

                    video_frames.append(frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                video_capture.release()

            return np.array(video_frames)
        else:
            # Use OpenCV’s VideoCapture to load the input video.
            video_capture = cv2.VideoCapture(video_filename)

            while video_capture.isOpened():

                ret, frame = video_capture.read()

                if not ret:
                    break

                video_frames.append(frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            video_capture.release()

            return np.array(video_frames)

    def _evaluate_video(self, video_frames: np.ndarray) -> np.ndarray | None:
        """
        mediapipe pose landmarking on the video frames

        Args:
            video_frames (np.ndarray): the video frames, shape (frame_len, height, width, 3)

        Returns:
            np.ndarray: the joints position, shape (frame_len, num_joints, 4)
        """

        joints_position = []

        with self.PoseLandmarker.create_from_options(self.options) as landmarker:

            frame_timestamp_ms = 0

            for i in range(video_frames.shape[0]):
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=video_frames[i]
                )

                pose_landmarker_result = landmarker.detect_for_video(
                    mp_image, frame_timestamp_ms
                )

                if (
                    pose_landmarker_result.pose_landmarks is None
                    or len(pose_landmarker_result.pose_landmarks) == 0
                ):
                    pose_landmarks = np.zeros((33, 4))

                    print(
                        f"{self.process_number} WARNING:: No pose landmarks detected in frame {i}"
                    )
                else:
                    pose_landmarks = np.array(
                        [
                            [
                                landmark.x,
                                landmark.y,
                                landmark.z,
                                landmark.visibility,
                            ]
                            for landmark in pose_landmarker_result.pose_landmarks[0]
                        ]
                    )

                joints_position.append(pose_landmarks)

                frame_timestamp_ms += int(1000 / 60)

        return np.array(joints_position)

    def build_features_targets(self, object_key):

        animation_frames = self._read_animation_frames(object_key)

        animation_name = object_key.split(self.path_split)[-1].split(".")[0]

        try:

            if self.bucket:
                video_path = f"videos/{animation_name}-30-0.avi"
            else:
                video_path = os.path.join(
                    object_key,
                    "..",
                    "..",
                    "videos",
                    f"{animation_name}-30-0.avi",
                )

                if not os.path.exists(video_path):
                    print(
                        f"{self.process_number} SKIPPING:: Key {video_path} video does not exist."
                    )
                    return None, None

            video_frames = self._read_video_frames(video_path)
        except oss2.exceptions.NoSuchKey:
            print(
                f"{self.process_number} SKIPPING:: Key {object_key} video does not exist."
            )
            return None, None

        assert len(video_frames) == len(
            animation_frames
        ), f"Frame data does not match the video length. {len(video_frames)} != {len(animation_frames)}, {animation_name}"

        if len(video_frames) < self.num_frames:
            print(
                f"{self.process_number} SKIPPING:: Animation {animation_name} has less than {self.num_frames} frames."
            )
            return None, None

        print(
            f"{self.process_number} INFO:: Read animation {animation_name}, total frames: {len(video_frames)}"
        )

        joints_position = self._evaluate_video(video_frames)

        # print(joints_position.shape, animation_frames.shape)

        # take every 30 frames, for the last 30 frames, take the last 30 frames
        features = []
        targets = []

        for i in range(0, len(joints_position), self.num_frames):

            end_frame = i + self.num_frames

            if end_frame > len(joints_position):
                start_frame = len(joints_position) - self.num_frames
                end_frame = len(joints_position)
            else:
                start_frame = end_frame - self.num_frames

            features.append(joints_position[start_frame:end_frame])
            targets.append(animation_frames[start_frame:end_frame])

        features = np.array(features)
        targets = np.array(targets)

        return features, targets

    def run(self):

        # counter = 0

        features = []
        targets = []

        for object_key in self.anim_euler_object_keys:

            animation_features, animation_targets = self.build_features_targets(
                object_key
            )

            # counter += 1

            # if counter > 1:
            #     break

        # print(animation_features.shape, animation_targets.shape)

        features = np.array(animation_features)
        targets = np.array(animation_targets)

        # print(features.shape, targets.shape)

        # get the first and the last animation name from self.anim_euler_object_keys
        first_animation_name = (
            self.anim_euler_object_keys[0].split(self.path_split)[-1].split(".")[0]
        )
        last_animation_name = (
            self.anim_euler_object_keys[-1].split(self.path_split)[-1].split(".")[0]
        )

        # put object to oss, under path "mediapipe-video-euler-data/"
        features_object_name = (
            f"features-{first_animation_name}-{last_animation_name}.npy"
        )
        targets_object_name = (
            f"targets-{first_animation_name}-{last_animation_name}.npy"
        )

        # save files to local
        np.save(features_object_name, features)
        np.save(targets_object_name, targets)

        if self.bucket:
            self.bucket.put_object(
                f"mediapipe-video-euler-data/{features_object_name}", features.tobytes()
            )
            self.bucket.put_object(
                f"mediapipe-video-euler-data/{targets_object_name}", targets.tobytes()
            )

        print(
            f"{self.process_number} INFO:: Put features to {features_object_name}, shape: {features.shape}, targets to {targets_object_name}, shape: {targets.shape}"
        )


if __name__ == "__main__":

    import time

    # get how many cpu cores available
    num_cores = os.cpu_count()

    # num_cores = 8 if num_cores > 8 else num_cores

    print(f"Using {num_cores} cores")

    if False:

        # 创建Server对象。
        # 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
        # 填写Bucket名称。
        endpoint = "oss-ap-southeast-1.aliyuncs.com"

        # 填写Bucket名称，并设置连接超时时间为30秒。
        bucket = oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)

        object_keys = []

        for obj in oss2.ObjectIterator(bucket, prefix="anim-euler-uniform/"):
            object_keys.append(obj.key)

        print(f"Total {len(object_keys)} animation files")

        # split the object_keys into num_cores parts, as evenly as possible
        object_keys_split = np.array_split(object_keys, num_cores)
    else:

        # anim_euler_dir = os.path.join("d:\\", "video2motion", "anim-euler-uniform")
        anim_euler_dir = os.path.join(
            os.path.expanduser("~"), "Documents", "video2motion", "anim-euler-uniform"
        )

        object_keys = list(os.listdir(anim_euler_dir))

        object_keys = [os.path.join(anim_euler_dir, obj) for obj in object_keys]

        print(f"Total {len(object_keys)} animation files")

        # split the object_keys into num_cores parts, as evenly as possible
        object_keys_split = np.array_split(object_keys, num_cores)

    processes = [
        MediapipeVideoEulerData(
            bucket=None,
            process_number=i,
            anim_euler_object_keys=object_keys,
            num_frames=30,
        )
        for i, object_keys in enumerate(object_keys_split)
    ]

    start_time = time.time()

    # run the process,
    for process in processes:
        process.start()

    for process in processes:
        # report the daemon attribute
        print(
            process.daemon,
            process.name,
            process.pid,
            process.exitcode,
            process.is_alive(),
        )

        process.join()

    end_time = time.time()

    print(f"Time taken: {end_time - start_time}")
