import json
import os
import glob

import numpy as np


def pad_data(json_data):

    longest = 0

    track_len = []

    longest_bone = []

    for k, v in json_data.items():

        times = v["times"]

        track_len.append((len(times), k))

        if len(times) > longest:

            longest = len(times)

            longest_bone = [k, times]

        flag = False

        for i in range(1, len(times)):

            step = times[i] - times[i - 1]

            if not np.isclose(step, 0.0166666, atol=0.00001):

                # print(step, k, i, times)
                # print(step, k, i)

                if step > 0.0166666 + 0.00001:
                    print(step)

                # flag = True

        # if flag:
        # return True

    # for tl, bone_name in track_len:

    #     if tl != 1 and tl < longest:

    #         print(f"padding {bone_name} from {tl} to {longest}")

    #         print(json_data[bone_name]["times"])
    #         print(longest_bone[1])

    #         break


if __name__ == "__main__":

    # anim_euler_dir = os.path.join(
    #     os.path.expanduser("~"), "Documents", "video2motion", "anim-json-euler"
    # )

    # anim_euler_files = glob.glob(os.path.join(anim_euler_dir, "*.json"))

    # anim_to_process = []

    # for anim_euler_file in anim_euler_files:

    #     print(f"{anim_euler_file}============================")

    #     with open(anim_euler_file, "r") as f:
    #         json_data = json.load(f)
    #         flag = pad_data(json_data)

    #         if flag:
    #             anim_to_process.append(anim_euler_file)

    # print(anim_to_process)

    # # save `anim_to_process` to a file
    # with open("anim_to_process.json", "w") as f:
    #     json.dump(anim_to_process, f)

    # break

    with open("anim_to_process.json", "r") as f:
        anim_to_process = json.load(f)

    for file_path in anim_to_process:

        with open(file_path, "r") as f:
            json_data = json.load(f)
            pad_data(json_data)

        # break
