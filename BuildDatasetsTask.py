from time import sleep
import time
import os
import json

import numpy as np
from multiprocessing import Process

from utils import extract_anim_euler_frames


# custom process class
class BuildDatasetsTask(Process):

    # override the constructor
    def __init__(
        self,
        queue_file_path,
        mediapipe_landmarks_path,
        resnet_features_path,
        resnet_queue_path,
        anim_euler_path,
        source="local",
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
        self.mediapipe_landmarks_path = mediapipe_landmarks_path
        self.resnet_features_path = resnet_features_path
        self.resnet_queue_path = resnet_queue_path
        self.anim_euler_path = anim_euler_path
        self.source = source
        self.size_limit = size_limit

        self.humanoid_name = "dors.glb"

    def read_resnet_data(self, animation_name, elevation, azimuth, n_frame):

        if self.source == "oss":

            file_path = f"{self.resnet_features_path}/{self.humanoid_name}/{animation_name}/{elevation}/{azimuth}/{n_frame}/feature.npy"

            return None
            # try:
            #     landmarks_obj = self.bucket.get_object(file_path)
            # except oss2.exceptions.NoSuchKey:
            #     print(f"oss2.exceptions.NoSuchKey: {file_path}")
            #     return

            # return json.loads(landmarks_obj.read())
        else:

            file_path = os.path.join(
                self.resnet_features_path,
                self.humanoid_name,
                animation_name,
                str(elevation),
                str(azimuth),
                str(n_frame),
                "feature.npy",
            )

            if not os.path.exists(file_path):
                print(f"file not found at path: {file_path}")
                return

            # with open(file_path, "r") as f:
            return np.load(file_path)

    def read_anim_euler_data(self, animation_name):

        if self.source == "oss":

            file_path = f"{self.anim_euler_path}/{animation_name}"
            return
            # try:
            #     anim_euler_obj = self.bucket.get_object(file_path)
            # except oss2.exceptions.NoSuchKey:
            #     print(f"oss2.exceptions.NoSuchKey: {file_path}")
            #     return

            # return json.loads(anim_euler_obj.read())
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

            mediapie_file_path = os.path.join(
                self.mediapipe_landmarks_path,
                self.humanoid_name,
                animation_name,
                str(elevation),
                str(azimuth),
                str(n_frame),
                "world_landmarks.json",
            )

            # check if mediapie_file_path is a file
            if not os.path.isfile(mediapie_file_path):
                print(f"skipping {mediapie_file_path}, mediapipe no results")
                continue

            resnet_res = self.read_resnet_data(
                animation_name, elevation, azimuth, n_frame
            )

            anim_euler = self.read_anim_euler_data(animation_name)

            bone_rotations = extract_anim_euler_frames(anim_euler, n_frame)

            features.append(resnet_res)
            targets.append(bone_rotations)

            if i and (i % 100 == 0):
                print(f"queue {queue_num} progress {i}/{len(queue_data)}")

        features = np.array(features)
        targets = np.array(targets)

        inputs_path = os.path.join(self.resnet_queue_path, "inputs")
        outputs_path = os.path.join(self.resnet_queue_path, "outputs")

        if not os.path.exists(inputs_path):
            os.makedirs(inputs_path)

        if not os.path.exists(outputs_path):
            os.makedirs(outputs_path)

        # save features and targets to npy file
        np.save(
            os.path.join(
                inputs_path,
                f"inputs_{queue_num}.npy",
            ),
            features,
        )

        np.save(
            os.path.join(
                outputs_path,
                f"outputs_{queue_num}.npy",
            ),
            targets,
        )


if __name__ == "__main__":

    pass
