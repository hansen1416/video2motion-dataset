from multiprocessing import Process
import os
import json

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

from constants import MEDIAPIPE_DIR
from utils import load_env_from_file

load_env_from_file()


class UploadMediapipeTask(Process):

    def __init__(self, queue_file_path) -> None:

        # execute the base constructor
        Process.__init__(self)

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

        humanoid_name = "dors.glb"

        for i, (animation_name, elevation, azimuth, n_frame) in enumerate(queue_data):

            object_name = f"mediapipe/{humanoid_name}/{animation_name}/{elevation}/{azimuth}/{n_frame}"

            local_result_dir = os.path.join(
                MEDIAPIPE_DIR,
                humanoid_name,
                animation_name,
                str(elevation),
                str(azimuth),
                str(n_frame),
            )

            if (
                not os.path.exists(os.path.join(local_result_dir, "landmarks.json"))
                or not os.path.exists(
                    os.path.join(local_result_dir, "world_landmarks.json")
                )
                or not os.path.exists(os.path.join(local_result_dir, "masked.jpg"))
            ):
                # output the local file path to local log
                with open("upload-mediapipe-error.log", "a") as f:
                    f.write(f"{local_result_dir}\n")
                print(f"queue {queue_num}, Skipping {local_result_dir}")
                continue

            # check if `object_name` already exists in the bucket
            if (
                self.bucket.object_exists(f"{object_name}/landmarks.json")
                and self.bucket.object_exists(f"{object_name}/world_landmarks.json")
                and self.bucket.object_exists(f"{object_name}/masked.jpg")
            ):
                print(f"queue {queue_num}, {object_name} already exists in the bucket")
                continue

            # 上传文件到OSS。
            # yourObjectName由包含文件后缀，不包含Bucket名称组成的Object完整路径，例如abc/efg/123.jpg。
            # yourLocalFile由本地文件路径加文件名包括后缀组成，例如/users/local/myfile.txt。
            result1 = self.bucket.put_object_from_file(
                f"{object_name}/landmarks.json",
                os.path.join(local_result_dir, "landmarks.json"),
            )
            result2 = self.bucket.put_object_from_file(
                f"{object_name}/world_landmarks.json",
                os.path.join(local_result_dir, "world_landmarks.json"),
            )
            result3 = self.bucket.put_object_from_file(
                f"{object_name}/masked.jpg",
                os.path.join(local_result_dir, "masked.jpg"),
            )

            if (
                int(result1.status) != 200
                or int(result2.status) != 200
                or int(result3.status) != 200
            ):
                # output the local file path to local log
                with open("upload-mediapipe-error.log", "a") as f:
                    f.write(f"{local_result_dir}\n")

                print(f"queue {queue_num}, Error uploading {local_result_dir}")

                continue

            if i % 1000 == 0:
                # HTTP返回码。
                print(
                    "queue {}, {} => {} http status: {}, {}, {}".format(
                        queue_num,
                        local_result_dir,
                        object_name,
                        result1.status,
                        result2.status,
                        result3.status,
                    )
                )


if __name__ == "__main__":

    pass
