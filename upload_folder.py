from multiprocessing import Process
import os
import sys
import time

import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider

from utils import load_env_from_file

load_env_from_file()


def split_array(array, n):
    """Splits an array into n pieces as evenly as possible.

    Args:
        array: The array to split.
        n: The number of pieces to split the array into.

    Returns:
        A list of lists, where each sublist is a piece of the original array.
    """
    chunk_size = len(array) // n  # Integer division for even floor
    remainder = len(array) % n

    pieces = []
    start = 0
    for i in range(n):
        end = start + chunk_size + (1 if i < remainder else 0)
        pieces.append(array[start:end])
        start = end

    return pieces


class UploadTask(Process):

    def __init__(self, file_list, oss_path="", process_idx="") -> None:

        # execute the base constructor
        Process.__init__(self)

        self.file_list = file_list
        self.oss_path = oss_path
        self.process_idx = process_idx

        # if oss_path is empty, take the last folder name of the first file
        if not oss_path:
            self.oss_path = os.path.basename(os.path.dirname(file_list[0])) + "/"

        # 使用环境变量中获取的RAM用户的访问密钥配置访问凭证。
        auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())

        # yourEndpoint填写Bucket所在地域对应的Endpoint。以华东1（杭州）为例，Endpoint填写为https://oss-cn-hangzhou.aliyuncs.com。
        endpoint = "oss-ap-southeast-1.aliyuncs.com"

        # 填写Bucket名称，并设置连接超时时间为30秒。
        self.bucket = oss2.Bucket(auth, endpoint, "pose-daten", connect_timeout=30)

    def run(self) -> None:
        """
        list all files in the upload_dir and upload them to oss, not checking sub directories
        """

        for filepath in self.file_list:
            if not os.path.isfile(filepath):
                continue

            filename = os.path.basename(filepath)

            target_path = f"{self.oss_path}{filename}"

            # check if the file already exists in oss
            if self.bucket.object_exists(target_path):
                print(f"{self.process_idx}: {target_path} already exists in oss")
                continue

            self.bucket.put_object_from_file(
                f"{target_path}",
                filepath,
            )

            print(f"{self.process_idx}: uploaded {filepath} to {target_path}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python upload_folder.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"{folder_path} does not exist")
        sys.exit(1)

    # get all files in the folder
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

    # get cpu count
    cpu_count = os.cpu_count()

    # split the files into cpu_count parts
    file_chunks = split_array(all_files, cpu_count)

    # create a process for each chunk
    processes = [
        UploadTask(
            file_list=chunk,
            process_idx=i,
        )
        for i, chunk in enumerate(file_chunks)
    ]

    # start the processes
    start_time = time.time()

    # run the process,
    for process in processes:
        process.start()

    for process in processes:
        # report the daemon attribute
        print(
            process.daemon,
            process.name,
            process.pid,
            process.exitcode,
            process.is_alive(),
        )

        process.join()

    end_time = time.time()

    print(f"Time taken: {end_time - start_time}")
