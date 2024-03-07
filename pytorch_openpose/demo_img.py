import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body

body_estimation = Body("model/body_pose_model.pth")

test_image = "images/18.jpg"
oriImg = cv2.imread(test_image)  # B,G,R order
candidate, subset = body_estimation(oriImg)

# the first two items of each row is the x, y coordinates of the body joint
joints_position = copy.deepcopy(candidate[:, :2])

image_width = oriImg.shape[1]
image_height = oriImg.shape[0]

# use the width and height of the image to normalize the joint positions, use image center as the origin
joints_position[:, 0] = (joints_position[:, 0] - image_width / 2) / image_width
joints_position[:, 1] = (joints_position[:, 1] - image_height / 2) / image_height

# now normalize the candidate

print(oriImg.shape)
print(candidate)
print(joints_position)
print(subset)
# exit()

canvas = copy.deepcopy(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)
# detect hand


plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis("off")
# plt.show()
