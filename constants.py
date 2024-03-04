import os

# from dotenv import load_dotenv


# # Load the environment variables from the .env file
# load_dotenv()


# # show env variable
# print(os.getenv("FROM_USB_DISK"))
# print("--------")


HOME_DIR = os.path.expanduser("~")

QUEUE_DIR = os.path.join(HOME_DIR, "Repos", "video2motion-screenshots", "queues")

ANIM_EULER_LOCAL_DIR = os.path.join(
    HOME_DIR,
    "Repos",
    "video2motion-screenshots",
    "anim-player",
    "public",
    "anim-json-euler",
)

if True:

    SCREENSHOT_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "screenshots")

    MEDIAPIPE_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "mediapipe")

    RESNET_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "resnet")

# else:

#     SCREENSHOT_DIR = os.path.join("D:\\", "video2motion", "screenshots")

#     MEDIAPIPE_DIR = os.path.join("D:\\", "video2motion", "mediapipe")

#     RESNET_DIR = os.path.join("D:\\", "Documents", "video2motion", "resnet")


if __name__ == "__main__":

    humanoid = "dors.glb"

    for f in os.listdir(os.path.join(SCREENSHOT_DIR, humanoid)):
        print(f)
        # os.remove(os.path.join(SCREEN_SHOT_DIR, f))
        break

    for q in os.listdir(QUEUE_DIR):
        print(q)
