import os
import json

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights


def extract_feature_vector(img_filename: str):
    # 2. Create a PyTorch Variable with the transformed image
    # Unsqueeze actually converts to a tensor by adding the number of images as another dimension.

    # print(input_image)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # model = resnet18(weights=ResNet18_Weights.DEFAULT)

    model.eval()

    input_image = Image.open(img_filename)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    t_img = input_batch

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    # my_embedding = torch.zeros(1, 512, 1, 1)
    my_embedding = torch.zeros(1, 2048, 1, 1)

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


if __name__ == "__main__":

    queue_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "queues",
    )

    screen_shot_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "video-recorder",
        "data",
    )

    humanoid_name = "dors.glb"

    # for f in os.listdir(os.path.join(screen_shot_dir, humanoid_name)):
    for q in os.listdir(queue_dir):

        with open(os.path.join(queue_dir, q)) as f:
            queue_data = json.load(f)

        for i, (animation_name, elevation, azimuth, n_frame) in enumerate(queue_data):
            screen_shot = os.path.join(
                screen_shot_dir,
                humanoid_name,
                animation_name,
                str(elevation),
                str(azimuth),
                f"{n_frame}.jpg",
            )

            # print(os.path.isfile(screen_shot))

            res = extract_feature_vector(screen_shot)

            print(res.shape)

            break

        break

    exit()

# model = torch.hub.load(
#     "pytorch/vision:v0.10.0", "resnet50", weights=ResNet50_Weights.DEFAULT
# )

# print(model)

current_dir = os.path.dirname(os.path.realpath(__file__))

filename = os.path.join(
    current_dir,
    "..",
    "video-recorder",
    "data",
    "dors.glb",
    "2hand Idle.json",
    "30",
    "0",
    "0.jpg",
)


# # print(input_batch.shape)


# with torch.no_grad():
#     output = model(input_batch)

# # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# # print(output.shape)

# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# # print(probabilities)


# res = extract_feature_vector(filename)

# print(res.shape)
