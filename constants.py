import os

HOME_DIR = os.path.expanduser("~")

QUEUE_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "queue")

SCREENSHOT_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "screenshots")

MEDIAPIPE_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "mediapipe")

RESNET_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "resnet")


if __name__ == "__main__":

    humanoid = "dors.glb"

    for f in os.listdir(os.path.join(SCREENSHOT_DIR, humanoid)):
        print(f)
        # os.remove(os.path.join(SCREEN_SHOT_DIR, f))

        break
