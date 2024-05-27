import os
import sys
import shutil

import cv2
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dotenv import load_dotenv

load_dotenv()


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

    def evaluate_video(self, video_filename):

        object_name = f"videos/{video_filename}"

        # bucket.get_object的返回值是一个类文件对象（File-Like Object），同时也是一个可迭代对象（Iterable）。
        # 填写Object的完整路径。Object完整路径中不能包含Bucket名称。
        object_stream = self.bucket.get_object(object_name)

        # content = object_stream.read()

        tmp_video_dir = os.path.join("tmp", "videos")

        # create the directory if it does not exist
        os.makedirs(tmp_video_dir, exist_ok=True)

        # print(sys.getsizeof(content))

        tmp_video_path = os.path.join(tmp_video_dir, video_filename)

        with open(tmp_video_path, "wb") as local_fileobj:
            shutil.copyfileobj(object_stream, local_fileobj)

        # Use OpenCV’s VideoCapture to load the input video.
        video_capture = cv2.VideoCapture(tmp_video_path)

        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        # You’ll need it to calculate the timestamp for each frame.
        fps = video_capture.get(cv2.CAP_PROP_FPS)

        # Loop through each frame in the video using VideoCapture#read()
        while video_capture.isOpened():
            num_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

            print(f"Number of frames: {num_frames}")

            # # Read the frame using VideoCapture#read()
            # ret, frame = video_capture.read()

            # # Break the loop if the video has ended
            # if not ret:
            #     break

            # # Convert the frame to RGB using cv2.COLOR_BGR2RGB
            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # # Convert the frame to a MediaPipe’s Image object
            # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # # Process the image using the MediaPipe instance
            # # results = mediapipe_instance.process(mp_image)

            # # Convert the MediaPipe results to a dictionary
            # # results_dict = python.solutions.drawing_utils._draw_landmarks(
            # #     frame, results.pose_landmarks, None, None, None, None
            # # )

            # # Draw the results on the frame using cv2.imshow()
            # # cv2.imshow("MediaPipe Pose", results_dict)

            # # Break the loop if the 'q' key is pressed
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        # # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        # mp_image = mp.Image(
        #     image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv
        # )

        video_capture.release()


if __name__ == "__main__":

    video_filename = "Aiming-30-0.avi"

    mediapie_video_evaludation = MediapieVideoEvaludation()
    mediapie_video_evaludation.evaluate_video(video_filename)

exit()

model_path = os.path.join("models", "pose_landmarker_lite.task")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
)

with PoseLandmarker.create_from_options(options) as landmarker:
    pass
