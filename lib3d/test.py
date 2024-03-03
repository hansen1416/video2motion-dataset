import numpy as np
import torch

from lib import vector_apply_euler_tensor
from lib1 import vector_apply_euler, Vector3, Euler

# generate random 3d vector

for _ in range(10000):
    vector = [0, 1, 0]
    euler = np.random.rand(3)

    v1 = vector_apply_euler_tensor(torch.Tensor(euler))
    v2 = vector_apply_euler(Vector3(*vector), Euler(*euler))

    v1 = v1.numpy()
    v2 = np.array([v2.x, v2.y, v2.z])

    # check if v1, v2 are the same
    if not np.allclose(v1, v2, atol=1e-5):
        print("v1", v1)
        print("v2", v2)
    else:
        # print("ok")
        pass
