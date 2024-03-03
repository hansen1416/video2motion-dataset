import os
import json

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()


# 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())


# yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
endpoint = "oss-ap-southeast-1.aliyuncs.com"

# 填写Bucket名称，并设置连接超时时间为30秒。
bucket = oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)

# print(bucket)


def upload_screen_shot():

    data_dir = os.path.join(os.path.dirname(__file__), "video-recorder", "data")
    humanoid_name = "dors.glb"

    queue_num = [4, 5, 6, 7]

    # for q in os.listdir(
    #     os.path.join(
    #         os.path.dirname(__file__), "video-recorder", "queue", humanoid_name
    #     )
    # ):
    for q in queue_num:

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "queues",
                f"queue{q}.json",
            )
        ) as f:
            queue_data = json.load(f)

        for animation_name, elevation, azimuth, n_frame in queue_data:

            object_name = f"screenshot/{humanoid_name}/{animation_name}/{elevation}/{azimuth}/{n_frame}.jpg"

            local_file = os.path.join(
                data_dir,
                humanoid_name,
                animation_name,
                str(elevation),
                str(azimuth),
                str(n_frame) + ".jpg",
            )

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
            print(
                "{} => {} http status: {}".format(
                    local_file, object_name, result.status
                )
            )

            if int(result.status) != 200:
                # output the local file path to local log
                with open("upload-screenshot-error.log", "a") as f:
                    f.write(f"{local_file}\n")


def upload_mediapipe():

    humanoid_name = "dors.glb"

    mediapipe_dir = os.path.join(
        os.path.dirname(__file__),
        "model-training",
        "mediapipe",
        "results",
        humanoid_name,
    )

    queue_num = [6, 7]

    # for q in os.listdir(
    #     os.path.join(
    #         os.path.dirname(__file__), "video-recorder", "queue", humanoid_name
    #     )
    # ):
    for q in queue_num:

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "queues",
                f"queue{q}.json",
            )
        ) as f:
            queue_data = json.load(f)

            for animation_name, elevation, azimuth, n_frame in queue_data:

                object_name = f"mediapipe/{humanoid_name}/{animation_name}/{elevation}/{azimuth}/{n_frame}"

                local_result_dir = os.path.join(
                    mediapipe_dir,
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
                    print(f"Skipping {local_result_dir}")
                    continue

                # check if `object_name` already exists in the bucket
                if (
                    bucket.object_exists(f"{object_name}/landmarks.json")
                    and bucket.object_exists(f"{object_name}/world_landmarks.json")
                    and bucket.object_exists(f"{object_name}/masked.jpg")
                ):
                    print(f"{object_name} already exists in the bucket")
                    continue

                # print(object_name, local_file)

                # 上传文件到OSS。
                # yourObjectName由包含文件后缀，不包含Bucket名称组成的Object完整路径，例如abc/efg/123.jpg。
                # yourLocalFile由本地文件路径加文件名包括后缀组成，例如/users/local/myfile.txt。
                result1 = bucket.put_object_from_file(
                    f"{object_name}/landmarks.json",
                    os.path.join(local_result_dir, "landmarks.json"),
                )
                result2 = bucket.put_object_from_file(
                    f"{object_name}/world_landmarks.json",
                    os.path.join(local_result_dir, "world_landmarks.json"),
                )
                result3 = bucket.put_object_from_file(
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

                # HTTP返回码。
                print(
                    "{} => {} http status: {}, {}, {}".format(
                        local_result_dir,
                        object_name,
                        result1.status,
                        result2.status,
                        result3.status,
                    )
                )


def upload_anim_data(sub_folder_name):

    pubnlic_dir = os.path.join(os.path.dirname(__file__), "anim-player", "public")

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


if __name__ == "__main__":

    # upload_screen_shot()

    upload_mediapipe()

    # upload_anim_data("anim-json")
    # upload_anim_data("anim-json-mixamo")
    # upload_anim_data("anim-json-euler")

    pass
