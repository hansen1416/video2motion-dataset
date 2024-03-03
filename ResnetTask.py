import json
import os
from multiprocessing import Process
from typing import Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

from constants import SCREENSHOT_DIR, RESNET_DIR


def extract_feature_vector(
    img_filename: str,
    preprocess: transforms.Compose,
    model: Union[resnet18, resnet50],
    model_name="resnet18",
):
    input_image = Image.open(img_filename)

    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    t_img = input_batch

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    if model_name == "resnet18":
        my_embedding = torch.zeros(1, 512, 1, 1)
    elif model_name == "resnet50":
        my_embedding = torch.zeros(1, 2048, 1, 1)
    else:
        raise ValueError("model must be resnet18 or resnet50")

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # Use the model object to select the desired layer
    layer = model._modules.get("avgpool")

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 6. Run the model on our transformed image
    model(t_img)

    # 7. Detach our copy function from the layer
    h.remove()

    # 8. Return the feature vector
    return my_embedding.squeeze().numpy()


class ResnetTask(Process):

    # override the constructor
    def __init__(
        self, queue_file_path, source="local", size_limit=None, model_name="resnet18"
    ):
        # execute the base constructor
        Process.__init__(self)

        self.queue_file_path = queue_file_path
        self.source = source
        self.size_limit = size_limit
        self.model_name = model_name

        self.humanoid = "dors.glb"

        # print(input_image)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if self.model_name == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif self.model_name == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        self.model.eval()

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

        for i, (animation_name, elevation, azimuth, n_frame) in enumerate(queue_data):

            imge_path = os.path.join(
                SCREENSHOT_DIR,
                self.humanoid,
                animation_name,
                str(elevation),
                str(azimuth),
                f"{n_frame}.jpg",
            )

            target_path = os.path.join(
                RESNET_DIR,
                self.humanoid,
                animation_name,
                str(elevation),
                str(azimuth),
                str(n_frame),
                "feature.npy",
            )

            if os.path.exists(target_path):
                print(f"resnet target path already exists: {target_path}")
                continue

            # print(os.path.exists(imge_path))
            data = extract_feature_vector(
                imge_path, self.preprocess, self.model, self.model_name
            )

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            np.save(target_path, data)

            if i and (i % 100 == 0):
                print(
                    f"queue {queue_num} progress {i}/{len(queue_data)}, results saved to {target_path}"
                )

        print(f"queue {queue_num} finished")
