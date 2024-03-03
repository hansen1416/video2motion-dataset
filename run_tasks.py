from time import sleep
import time
import os
import json

from tqdm import tqdm
import numpy as np
from multiprocessing import Process
from dotenv import load_dotenv

from constants import (
    SCREENSHOT_DIR,
    QUEUE_DIR,
    MEDIAPIPE_DIR,
    RESNET_DIR,
    ANIM_EULER_LOCAL_DIR,
)
from ResnetTask import ResnetTask


# Load the environment variables from the .env file
load_dotenv()


if __name__ == "__main__":

    queue_files = [os.path.join(QUEUE_DIR, q) for q in os.listdir(QUEUE_DIR)]

    processes = [
        ResnetTask(
            queue_file_path=q,
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
