from dotenv import load_dotenv

load_dotenv()


if __name__ == "__main__":

    import os
    from oss2_uploader import folder_downloader

    folder_downloader(
        "pose-daten",
        "oss-ap-southeast-1.aliyuncs.com",
        "videos",
        os.path.join(os.path.expanduser("~"), "Documents", "video2motion", "videos"),
    )
