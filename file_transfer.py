import os
from oss2_uploader import folder_downloader
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    # folder_downloader(
    #     bucket_name="pose-daten",
    #     oss_endpoint="oss-ap-southeast-1.aliyuncs.com",
    #     oss_prefix="detectron2d",
    #     target_dir=os.path.join(
    #         os.path.expanduser("~"), "Documents", "video2motion", "detectron2d"
    #     ),
    # )

    folder_downloader(
        bucket_name="pose-daten",
        oss_endpoint="oss-ap-southeast-1.aliyuncs.com",
        oss_prefix="results3d",
        target_dir=os.path.join(
            os.path.expanduser("~"), "Documents", "video2motion", "results3d"
        ),
    )
