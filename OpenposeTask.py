import json
import os
import copy
from multiprocessing import Process

import shutil
import numpy as np
import cv2
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

# import matplotlib.pyplot as plt

from pytorch_openpose.src.body import Body
from utils import load_env_from_file

# from pytorch_openpose.src import util


load_env_from_file()


def openpose_predict(file_path: str):

    model_path = os.path.join(
        os.path.dirname(__file__), "pytorch_openpose", "model", "body_pose_model.pth"
    )

    body_estimation = Body(model_path)

    # print(body_estimation)

    oriImg = cv2.imread(file_path)  # B,G,R order
    candidate, subset = body_estimation(oriImg)

    if candidate is None or candidate.shape[0] == 0:
        return None

    # the first two items of each row is the x, y coordinates of the body joint
    joints_position = copy.deepcopy(candidate[:, :2])

    image_width = oriImg.shape[1]
    image_height = oriImg.shape[0]

    # use the width and height of the image to normalize the joint positions, use image center as the origin
    joints_position[:, 0] = (joints_position[:, 0] - image_width / 2) / image_width
    joints_position[:, 1] = (joints_position[:, 1] - image_height / 2) / image_height

    joints_position = np.array(joints_position)

    # print(joints_position.shape)

    return joints_position

    # canvas = copy.deepcopy(oriImg)
    # canvas = util.draw_bodypose(canvas, candidate, subset)
    # # detect hand

    # plt.imshow(canvas[:, :, [2, 1, 0]])
    # plt.axis("off")
    # plt.show()


class OpenposeTask(Process):

    # override the constructor
    def __init__(self, queue_file_path, size_limit=None):
        self.queue_file_path = queue_file_path

        # 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

        # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
        endpoint = "oss-ap-southeast-1.aliyuncs.com"

        # 填写Bucket名称，并设置连接超时时间为30秒。
        self.bucket = oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)

    def run(self) -> None:

        with open(self.queue_file_path, "r") as f:
            queue_data = json.load(f)

        queue_num = int(
            os.path.basename(self.queue_file_path)
            .replace(".json", "")
            .replace("queue", "")
        )

        if self.size_limit:
            # for testing, only get 100
            queue_data = queue_data[: self.size_limit]

        humanoid_name = "dors.glb"

        for i, (animation_name, elevation, azimuth, n_frame) in enumerate(queue_data):

            object_name = f"openpose/{humanoid_name}/{animation_name}/{elevation}/{azimuth}/{n_frame}/j.npy"

            # check if `object_name` already exists in the bucket
            if self.bucket.object_exists(object_name):
                print(f"queue {queue_num}, {object_name} already exists in the bucket")
                continue

            # get oss screenshot image path, then save it to local tmp file
            screenshot_path = f"screenshot/{humanoid_name}/{animation_name}/{elevation}/{azimuth}/{n_frame}.jpg"
            file_stream = self.bucket.get_object(screenshot_path)

            with open("temp.jpg", "wb") as f:
                shutil.copyfileobj(file_stream, f)
            # prediction
            joints_position = openpose_predict("temp.jpg")

            if joints_position is None:

                exmpty_object_name = f"openpose/{humanoid_name}/{animation_name}/{elevation}/{azimuth}/{n_frame}/empty.txt"

                result = self.bucket.put_object(exmpty_object_name, "")
                continue

            # convert the result from `openpose_predict` to bytes
            joints_position_bytes = joints_position.tobytes()

            # upload bytes to oss
            result = self.bucket.put_object_from_file(
                object_name, joints_position_bytes
            )

            if int(result.status) != 200:
                # output the local file path to local log
                with open("upload-resnet-error.log", "a") as f:
                    f.write(f"{object_name}\n")

                print(
                    "queue {}, Upload failed! {} http status: {}".format(
                        queue_num, object_name, result.status
                    )
                )

            if i % 1000 == 0:
                # HTTP返回码。
                print(
                    "queue {}, {} http status: {}".format(
                        queue_num, object_name, result.status
                    )
                )


if __name__ == "__main__":

    from constants import SCREENSHOT_DIR

    humanoid = "dors.glb"

    for animation_name in os.listdir(os.path.join(SCREENSHOT_DIR, humanoid)):

        image_path = os.path.join(
            SCREENSHOT_DIR, humanoid, animation_name, "30", "0", "0.jpg"
        )

        if os.path.exists(image_path):
            print(f"File not found: {image_path}")

            openpose_predict(image_path)

        break
