import os
import sys
import shutil
import random
import string
import json

import cv2
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
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


class MediapieVideoEvaludation:

    def __init__(self) -> None:
        # 创建Server对象。
        # 从环境变量中获取访问凭证。运行本代码示例之前，请确保已设置环境变量OSS_ACCESS_KEY_ID和OSS_ACCESS_KEY_SECRET。
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
        # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
        # 填写Bucket名称。
        endpoint = "oss-ap-southeast-1.aliyuncs.com"

        # 填写Bucket名称，并设置连接超时时间为30秒。
        self.bucket = oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)

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
            animation_name = obj.key.split("/")[-1].split(".")[0]

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

    def anim_data_loader(self, json_data, start_frame=0, end_frame=30):
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

            # we must consider the time, because the time step is not always 0.166666 ms

            # print(
            #     "values.shape",
            #     values.shape,
            #     start_frame,
            #     end_frame,
            #     values[start_frame:end_frame, :].shape,
            # )

            if start_frame == end_frame:
                data_bones.append(values[start_frame, :])
            else:
                data_bones.append(values[start_frame:end_frame, :])

        # now the bones shape is (num_bones, frame_len, 3)
        data_bones = np.array(data_bones)

        # print(data_bones.shape)

        # convert it to frame wise data, shape (frame_len, num_bones, 3)
        data_bones_frame = np.transpose(data_bones, (1, 0, 2))

        # print(data_bones_frame.shape)

        return data_bones_frame

    def evaluate_video(self):

        counter = 0

        for obj in oss2.ObjectIterator(self.bucket, prefix="anim-euler-uniform/"):
            animation_name = obj.key.split("/")[-1].split(".")[0]

            with OSSObjectTmpReader(obj.key, self.bucket) as tmp_file:
                with open(tmp_file, "r") as f:
                    json_data = json.load(f)

                    frame_lenth = len(json_data["Hips"])

                    print(
                        f"Read animation {animation_name}, frame length: {frame_lenth}"
                    )

            video_frames = []

            with OSSObjectTmpReader(
                f"videos/{animation_name}-30-0.avi", self.bucket
            ) as tmp_video_file:

                # Use OpenCV’s VideoCapture to load the input video.
                video_capture = cv2.VideoCapture(tmp_video_file)

                num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

                print(f"Read video {animation_name}, total frames: {num_frames}")

                while video_capture.isOpened():

                    ret, frame = video_capture.read()

                    if not ret:
                        break

                    video_frames.append(frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                video_capture.release()

            assert (
                frame_lenth == num_frames
            ), f"Frame length does not match the video length. {frame_lenth} != {num_frames}, {animation_name}"

            assert (
                len(video_frames) == frame_lenth
            ), f"Frame data does not match the video length. {len(video_frames)} != {frame_lenth}, {animation_name}"

            video_frames = np.array(video_frames)

            # print(video_frames.shape)

            # print(json_data["Hips"])

            with self.PoseLandmarker.create_from_options(self.options) as landmarker:

                frame_timestamp_ms = 0

                for i in range(video_frames.shape[0]):
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=video_frames[i]
                    )

                    pose_landmarker_result = landmarker.detect_for_video(
                        mp_image, frame_timestamp_ms
                    )

                    pose_landmarks = np.array(
                        [
                            [landmark.x, landmark.y, landmark.z, landmark.visibility]
                            for landmark in pose_landmarker_result.pose_landmarks[0]
                        ]
                    )

                    print(pose_landmarks.shape)

                    frame_timestamp_ms += int(1000 / 60)

                    break

            counter += 1

            if counter > 0:
                break


if __name__ == "__main__":

    video_filename = "Aiming"

    mediapie_video_evaludation = MediapieVideoEvaludation()
    # mediapie_video_evaludation.check_frames(limit=100)

    mediapie_video_evaludation.evaluate_video()
