import os
from oss2_uploader import folder_downloader, folder_uploader
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":

    # folder_downloader(
    #     bucket_name="pose-daten",
    #     oss_endpoint="oss-ap-southeast-1.aliyuncs.com",
    #     oss_prefix="anim-json/",
    #     target_dir=os.path.join(
    #         os.path.expanduser("~"),
    #         "Documents",
    #         "video2motion-animsaver",
    #         "data",
    #     ),
    # )

    folder_uploader(
        folder_path=os.path.join(
            os.path.expanduser("~"),
            "Documents",
            "video2motion-animplayer",
            "public",
            "anim-euler-uniform",
        ),
        bucket_name="pose-daten",
        oss_endpoint="oss-ap-southeast-1.aliyuncs.com",
        oss_path="anim-euler-uniform/",
    )
