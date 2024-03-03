import math
import numpy as np
import torch


def quaternion_length(q: np.ndarray) -> float:
    return math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    l = quaternion_length(q)

    if l == 0:
        q[0] = 0
        q[1] = 0
        q[2] = 0
        q[3] = 1
    else:
        l = 1 / l
        q[0] *= l
        q[1] *= l
        q[2] *= l
        q[3] *= l

    return q


def quaternion_from_euler_arr(euler: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """
    save logic as quaternion_from_euler, but for numpy array
    """
    x = euler[0]
    y = euler[1]
    z = euler[2]

    c1 = math.cos(x / 2)
    c2 = math.cos(y / 2)
    c3 = math.cos(z / 2)

    s1 = math.sin(x / 2)
    s2 = math.sin(y / 2)
    s3 = math.sin(z / 2)

    quaternion = np.zeros(4)

    if order == "XYZ":
        quaternion[0] = s1 * c2 * c3 + c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 - s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 + s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "YXZ":
        quaternion[0] = s1 * c2 * c3 + c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 - s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 - s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 + s1 * s2 * s3
    elif order == "ZXY":
        quaternion[0] = s1 * c2 * c3 - c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 + s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 + s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "ZYX":
        quaternion[0] = s1 * c2 * c3 - c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 + s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 - s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 + s1 * s2 * s3
    elif order == "YZX":
        quaternion[0] = s1 * c2 * c3 + c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 + s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 - s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "XZY":
        quaternion[0] = s1 * c2 * c3 - c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 - s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 + s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 + s1 * s2 * s3
    else:
        raise ValueError(f"Unknown order: {order}")

    return normalize_quaternion(quaternion)


def vector_apply_quaternion_arr(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    vx = v[0]
    vy = v[1]
    vz = v[2]

    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)

    v[0] = vx + qw * tx + qy * tz - qz * ty
    v[1] = vy + qw * ty + qz * tx - qx * tz
    v[2] = vz + qw * tz + qx * ty - qy * tx

    return v


def vector_apply_euler_arr(
    v: np.ndarray, euler: np.ndarray, order: str = "XYZ"
) -> np.ndarray:
    q = quaternion_from_euler_arr(euler, order)
    return vector_apply_quaternion_arr(v, q)


def quaternion_from_euler_tensor(
    euler: torch.Tensor, order: str = "XYZ"
) -> torch.Tensor:
    """
    save logic as quaternion_from_euler, but for torch tensor
    """
    x = euler[0]
    y = euler[1]
    z = euler[2]

    c1 = torch.cos(x / 2)
    c2 = torch.cos(y / 2)
    c3 = torch.cos(z / 2)

    s1 = torch.sin(x / 2)
    s2 = torch.sin(y / 2)
    s3 = torch.sin(z / 2)

    quaternion = torch.zeros(4)

    if order == "XYZ":
        quaternion[0] = s1 * c2 * c3 + c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 - s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 + s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "YXZ":
        quaternion[0] = s1 * c2 * c3 + c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 - s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 - s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 + s1 * s2 * s3
    elif order == "ZXY":
        quaternion[0] = s1 * c2 * c3 - c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 + s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 + s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "ZYX":
        quaternion[0] = s1 * c2 * c3 - c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 + s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 - s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 + s1 * s2 * s3
    elif order == "YZX":
        quaternion[0] = s1 * c2 * c3 + c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 + s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 - s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "XZY":
        quaternion[0] = s1 * c2 * c3 - c1 * s2 * s3
        quaternion[1] = c1 * s2 * c3 - s1 * c2 * s3
        quaternion[2] = c1 * c2 * s3 + s1 * s2 * c3
        quaternion[3] = c1 * c2 * c3 + s1 * s2 * s3
    else:
        raise ValueError(f"Unknown order: {order}")

    return normalize_quaternion(quaternion)


def vector_apply_quaternion_tensor(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    vx = v[0]
    vy = v[1]
    vz = v[2]

    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]

    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)

    v[0] = vx + qw * tx + qy * tz - qz * ty
    v[1] = vy + qw * ty + qz * tx - qx * tz
    v[2] = vz + qw * tz + qx * ty - qy * tx

    return v


def vector_apply_euler_tensor(
    euler: torch.Tensor,
) -> torch.Tensor:
    # assume up vector is [0, 1, 0]
    v = torch.Tensor([0, 1, 0])
    order = "XYZ"

    q = quaternion_from_euler_tensor(euler, order)
    return vector_apply_quaternion_tensor(v, q)


if __name__ == "__mian__":

    v1 = vector_apply_euler_tensor(np.array([1, 0, 0]), np.array([0, 0, 0]))

    print(v1)
