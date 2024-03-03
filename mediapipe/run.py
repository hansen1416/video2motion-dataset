import json
import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def save_pose_visualize_image(annotated_image, image_name="tmp.jpg"):
    bgr_array = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(image_name, bgr_array)


def save_mask_image(segmentation_mask):
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255

    cv2.imwrite("tmp_mask.jpg", visualized_mask)


def save_pose_results(mp_image, pose_landmarker_result, res_dir):

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    landmark_filename = os.path.join(res_dir, "landmarks.json")
    world_landmark_filename = os.path.join(res_dir, "world_landmarks.json")
    masked_filename = os.path.join(res_dir, "masked.jpg")

    if (
        os.path.exists(landmark_filename)
        and os.path.exists(world_landmark_filename)
        and os.path.exists(masked_filename)
    ):
        print(f"Skipping {res_dir}")
        return

    pose_landmarks = pose_landmarker_result.pose_landmarks
    pose_world_landmarks = pose_landmarker_result.pose_world_landmarks
    segmentation_masks = pose_landmarker_result.segmentation_masks

    if len(pose_landmarks) == 0 or len(pose_world_landmarks) == 0:
        print(f"No pose predicted on {res_dir}")
        return

    # save pose landmarks as json
    pose_landmarks_json = []

    for lm in pose_landmarks[0]:
        pose_landmarks_json.append(
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence,
            }
        )

    with open(landmark_filename, "w") as jsonfile:
        json.dump(pose_landmarks_json, jsonfile, indent=4)

    # save world pose landmarks as json
    pose_world_landmarks_json = []

    for lm in pose_world_landmarks[0]:
        pose_world_landmarks_json.append(
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence,
            }
        )

    with open(world_landmark_filename, "w") as jsonfile:
        json.dump(pose_world_landmarks_json, jsonfile, indent=4)

    segmentation_mask_np = segmentation_masks[0].numpy_view().copy()
    # Threshold for small values
    threshold = 1e-2
    # set where the mask is less than threshold to 0
    segmentation_mask_np[segmentation_mask_np > threshold] = 1

    # cast to uint8
    segmentation_mask_np = segmentation_mask_np.astype(np.uint8)

    # use `segmentation_mask_np` as mask to select image data from `mp_image.numpy_view()`
    # and save it as a new image
    # save pose image
    pose_image = mp_image.numpy_view().copy()

    pose_image[segmentation_mask_np == 0, :] = 0
    # set pose_image where its value is close to [0, 255, 0] to [0, 0, 0]
    # pose_image[np.all(pose_image == [0, 255, 0], axis=2)] = [
    #     0,
    #     0,
    #     0,
    # ]

    # print(pose_image)

    cv2.imwrite(masked_filename, pose_image)

    # print(pose_landmarks)
    # print(pose_landmarks.landmark)
    # print(pose_world_landmarks.landmark)
    # print(segmentation_masks[0].numpy_view())
    # print(mp_image.numpy_view().shape)

    return
    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(
        mp_image.numpy_view(), pose_landmarker_result
    )
    save_pose_visualize_image(annotated_image)

    segmentation_mask = segmentation_masks[0].numpy_view()
    save_mask_image(segmentation_mask)


# `models` dir in the current file directory
model_path = os.path.join(
    os.path.dirname(__file__), "models", "pose_landmarker_heavy.task"
)

source_image_dir = os.path.join(
    os.path.dirname(__file__), "..", "..", "video-recorder", "data"
)

queue_dir = os.path.join(os.path.dirname(__file__), "..", "..", "queues")

charater_names = ["dors.glb"]

results_dir = os.path.join(os.path.dirname(__file__), "results")

# prepare mediapipe settings
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the video mode:
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    output_segmentation_masks=True,
)


with PoseLandmarker.create_from_options(options) as landmarker:

    for char in charater_names:

        for queue_num in [6, 7]:

            queue_file = os.path.join(queue_dir, f"queue{queue_num}.json")

            with open(queue_file) as f:
                queue_data = json.load(f)

                for task in queue_data:

                    # result dir for current pose image
                    res_dir = os.path.join(results_dir, char, *list(map(str, task)))

                    # check if results already exists
                    if os.path.exists(res_dir) and len(os.listdir(res_dir)) == 3:
                        print(f"Skipping {res_dir}")
                        continue

                    image_file = os.path.join(
                        source_image_dir, char, *list(map(str, task))
                    )

                    image_file += ".jpg"

                    if os.path.isfile(image_file):
                        print(f"Processing {image_file}")

                        # Load the input image from an image file.
                        mp_image = mp.Image.create_from_file(image_file)
                        # Perform pose landmarking on the provided single image.
                        # The pose landmarker must be created with the image mode.
                        pose_landmarker_result = landmarker.detect(mp_image)

                        save_pose_results(mp_image, pose_landmarker_result, res_dir)

                    # break
            # break
