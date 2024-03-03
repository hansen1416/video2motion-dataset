from time import sleep
import time
import os
import json

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
from tqdm import tqdm
import numpy as np
from multiprocessing import Process
from dotenv import load_dotenv

from utils import get_landmarks1d, extract_anim_euler_frames

# Load the environment variables from the .env file
load_dotenv()


# custom process class
class CustomProcess(Process):

    # override the constructor
    def __init__(
        self,
        queue_file_path,
        mediapipe_path,
        anim_euler_path,
        source="oss",
        size_limit=None,
    ):
        # execute the base constructor
        Process.__init__(self)

        # An instance of the multiprocessing.Value can be defined in the constructor of a custom class as a shared instance variable.
        # The constructor of the multiprocessing.Value class requires that we specify the data type and an initial value.
        # We can define an instance attribute as an instance of the multiprocessing.Value
        # which will automatically and correctly be shared between processes.
        # initialize integer attribute
        # self.data = Value("i", 0)

        self.queue_file_path = queue_file_path
        self.mediapipe_path = mediapipe_path
        self.anim_euler_path = anim_euler_path
        self.source = source
        self.size_limit = size_limit

        if self.source == "oss":
            # oss auth
            auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

            endpoint = "oss-ap-southeast-1.aliyuncs.com"

            # oss bucket, timeout 30s
            self.bucket = oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)

    def read_mediapipe_data(self, animation_name, elevation, azimuth, n_frame):

        if self.source == "oss":

            file_path = f"{self.mediapipe_path}/{animation_name}/{elevation}/{azimuth}/{n_frame}/world_landmarks.json"

            try:
                landmarks_obj = self.bucket.get_object(file_path)
            except oss2.exceptions.NoSuchKey:
                print(f"oss2.exceptions.NoSuchKey: {file_path}")
                return

            return json.loads(landmarks_obj.read())
        else:

            file_path = os.path.join(
                self.mediapipe_path,
                animation_name,
                str(elevation),
                str(azimuth),
                str(n_frame),
                "world_landmarks.json",
            )

            if not os.path.exists(file_path):
                print(f"file not found at path: {file_path}")
                return

            with open(file_path, "r") as f:
                return json.load(f)

    def read_anim_euler_data(self, animation_name):

        if self.source == "oss":

            file_path = f"{self.anim_euler_path}/{animation_name}"

            try:
                anim_euler_obj = self.bucket.get_object(file_path)
            except oss2.exceptions.NoSuchKey:
                print(f"oss2.exceptions.NoSuchKey: {file_path}")
                return

            return json.loads(anim_euler_obj.read())
        else:

            file_path = os.path.join(self.anim_euler_path, animation_name)

            if not os.path.exists(file_path):
                print(f"file not found at path: {file_path}")
                return

            with open(file_path, "r") as f:
                return json.load(f)

    # override the run function
    def run(self):

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

        features = []
        targets = []

        for i, (animation_name, elevation, azimuth, n_frame) in enumerate(queue_data):

            world_landmarks = self.read_mediapipe_data(
                animation_name, elevation, azimuth, n_frame
            )

            anim_euler = self.read_anim_euler_data(animation_name)

            if not world_landmarks or not anim_euler:
                print(
                    f"skipping {animation_name} {elevation} {azimuth} {n_frame}, empty data"
                )
                continue

            landmarks1d = get_landmarks1d(world_landmarks)

            features.append(landmarks1d)

            # print(landmarks1d)

            bone_rotations = extract_anim_euler_frames(anim_euler, n_frame)

            targets.append(bone_rotations)

            # print(bone_rotations)

            if i and (i % 100 == 0):
                print(f"queue {queue_num} progress {i}/{len(queue_data)}")

        features = np.array(features)
        targets = np.array(targets)

        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "data", "inputs")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "data", "inputs"))

        if not os.path.exists(
            os.path.join(os.path.dirname(__file__), "data", "outputs")
        ):
            os.makedirs(os.path.join(os.path.dirname(__file__), "data", "outputs"))

        # save features and targets to npy file
        np.save(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "inputs",
                f"inputs_{queue_num}.npy",
            ),
            features,
        )

        np.save(
            os.path.join(
                os.path.dirname(__file__),
                "data",
                "outputs",
                f"outputs_{queue_num}.npy",
            ),
            targets,
        )


if __name__ == "__main__":

    queue_nums = [0, 1, 2, 3, 4, 5, 6, 7]

    # Number of processes to use

    queue_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "queues",
    )

    humanoid_name = "dors.glb"
    mediapipe_path = f"mediapipe/{humanoid_name}"
    anim_euler_path = f"anim-json-euler"

    mediapipe_path_local = os.path.join(
        os.path.dirname(__file__), "..", "mediapipe", "results", humanoid_name
    )
    anim_euler_path_local = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "anim-player",
        "public",
        "anim-json-euler",
    )

    processes = [
        CustomProcess(
            queue_file_path=os.path.join(queue_dir, f"queue{i}.json"),
            mediapipe_path=mediapipe_path_local,
            anim_euler_path=anim_euler_path_local,
            source="local",
            size_limit=None,
        )
        for i in queue_nums
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
