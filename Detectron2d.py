import os

import numpy as np


def decode(filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    print("Processing {}".format(filename))
    data = np.load(filename, encoding="latin1", allow_pickle=True)
    bb = data["boxes"]
    kp = data["keypoints"]
    metadata = data["metadata"].item()
    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            # No bbox/keypoints detected for this frame -> will be interpolated
            results_bb.append(
                np.full(4, np.nan, dtype=np.float32)
            )  # 4 bounding box coordinates
            results_kp.append(
                np.full((17, 4), np.nan, dtype=np.float32)
            )  # 17 COCO keypoints
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        best_bb = bb[i][1][best_match, :4]
        best_kp = kp[i][1][best_match].T.copy()
        results_bb.append(best_bb)
        results_kp.append(best_kp)

    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)
    kp = kp[:, :, :2]  # Extract (x, y)

    # Fix missing bboxes/keypoints by linear interpolation
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])

    print("{} total frames processed".format(len(bb)))
    print("{} frames were interpolated".format(np.sum(~mask)))
    print("----------")

    return [
        {
            "start_frame": 0,  # Inclusive
            "end_frame": len(kp),  # Exclusive
            "bounding_boxes": bb,
            "keypoints": kp,
        }
    ], metadata


data_dir = os.path.join(
    os.path.expanduser("~"), "Documents", "video2motion", "detectron2d"
)

# filenames = os.listdir(data_dir)


# print(filenames)

# for filename in filenames:

#     data = np.load(
#         os.path.join(data_dir, filename), encoding="latin1", allow_pickle=True
#     )

#     print(data)

#     break


filename = "Zombie Turn-30-0.avi.npz"

res, metadata = decode(os.path.join(data_dir, filename))

print(res)
print(metadata)

# print(kp)
