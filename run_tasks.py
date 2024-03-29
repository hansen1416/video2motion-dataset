from time import sleep
import time
import os

from constants import (
    QUEUE_DIR,
    MEDIAPIPE_DIR,
    RESNET_DIR,
    ANIM_EULER_LOCAL_DIR,
)
from ResnetTask import ResnetTask
from OpenposeTask import OpenposeTask
from BuildDatasetsTask import BuildDatasetsTask
from UploadScreenshotTask import UploadScreenshotTask
from UploadMediapipeTask import UploadMediapipeTask
from UploadResnetTask import UploadResnetTask


if __name__ == "__main__":

    mediapipe_landmarks_path = os.path.join(MEDIAPIPE_DIR, "landmarks")

    resnet_features_path = os.path.join(RESNET_DIR, "features")

    resnet_queue_path = os.path.join(RESNET_DIR, "queue_data")

    queue_files = [os.path.join(QUEUE_DIR, q) for q in os.listdir(QUEUE_DIR)]

    if False:
        processes = [
            ResnetTask(
                queue_file_path=q,
            )
            for q in queue_files
        ]
    elif False:
        processes = [
            ResnetBuildTask(
                queue_file_path=q,
                mediapipe_landmarks_path=mediapipe_landmarks_path,
                resnet_features_path=resnet_features_path,
                resnet_queue_path=resnet_queue_path,
                anim_euler_path=ANIM_EULER_LOCAL_DIR,
                # size_limit=10,
            )
            for q in queue_files
        ]
    elif False:
        processes = [
            UploadScreenshotTask(
                queue_file_path=q,
            )
            for q in queue_files
        ]
    elif False:
        processes = [
            UploadMediapipeTask(
                queue_file_path=q,
            )
            for q in queue_files
        ]
    elif False:
        processes = [
            UploadResnetTask(
                queue_file_path=q,
            )
            for q in queue_files
        ]
    elif True:
        processes = [
            OpenposeTask(
                queue_file_path=q,
                # size_limit=4,
            )
            for q in queue_files
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
