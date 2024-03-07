import json
import os
from multiprocessing import Process

from pytorch_openpose.src.body import Body


def openpose_predict():

    print(Body)

    pass


class ResnetTask(Process):

    # override the constructor
    def __init__(
        self, queue_file_path, source="local", size_limit=None, model_name="resnet18"
    ):
        self.queue_file_path = queue_file_path

    def run(self) -> None:

        with open(self.queue_file_path, "r") as f:
            queue_data = json.load(f)

        queue_num = int(
            os.path.basename(self.queue_file_path)
            .replace(".json", "")
            .replace("queue", "")
        )


if __name__ == "__main__":

    openpose_predict()
