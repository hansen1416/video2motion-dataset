import math
import os
import json


class Euler:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def to_array(self):
        return [self.x, self.y, self.z]

    def __str__(self) -> str:
        return str([self.x, self.y, self.z])


class Vector3:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self) -> str:
        return str([self.x, self.y, self.z])


class Quaternion:
    def __init__(self, x=0, y=0, z=0, w=1):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __str__(self) -> str:
        return str([self.x, self.y, self.z, self.w])

    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def normalize(self):
        l = self.length()

        if l == 0:
            self.x = 0
            self.y = 0
            self.z = 0
            self.w = 1
        else:
            l = 1 / l
            self.x *= l
            self.y *= l
            self.z *= l
            self.w *= l


class Matrix4:
    def __init__(self):
        self.elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def matrix_rotation_from_quaternion(
    position: Vector3 = Vector3(0, 0, 0),
    quaternion: Quaternion = Quaternion(0, 0, 0, 1),
    scale: Vector3 = Vector3(1, 1, 1),
):
    quaternion.normalize()

    matrix = Matrix4()
    te = matrix.elements

    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    x2 = x + x
    y2 = y + y
    z2 = z + z

    xx = x * x2
    xy = x * y2
    xz = x * z2

    yy = y * y2
    yz = y * z2
    zz = z * z2

    wx = w * x2
    wy = w * y2
    wz = w * z2

    sx = scale.x
    sy = scale.y
    sz = scale.z

    te[0] = (1 - (yy + zz)) * sx
    te[1] = (xy + wz) * sx
    te[2] = (xz - wy) * sx
    te[3] = 0

    te[4] = (xy - wz) * sy
    te[5] = (1 - (xx + zz)) * sy
    te[6] = (yz + wx) * sy
    te[7] = 0

    te[8] = (xz + wy) * sz
    te[9] = (yz - wx) * sz
    te[10] = (1 - (xx + yy)) * sz
    te[11] = 0

    te[12] = position.x
    te[13] = position.y
    te[14] = position.z
    te[15] = 1

    return matrix


def clamp(value, min_value, max_value):
    """
    function clamp( value, min, max ) {

            return Math.max( min, Math.min( max, value ) );

    }
    """
    return max(min_value, min(max_value, value))


def euler_from_matrix(matrix: Matrix4, order: str = "XYZ"):

    te = matrix.elements
    m11 = te[0]
    m12 = te[4]
    m13 = te[8]
    m21 = te[1]
    m22 = te[5]
    m23 = te[9]
    m31 = te[2]
    m32 = te[6]
    m33 = te[10]

    if order == "XYZ":
        y = math.asin(clamp(m13, -1, 1))

        if abs(m13) < 0.9999999:
            x = math.atan2(-m23, m33)
            z = math.atan2(-m12, m11)
        else:
            x = math.atan2(m32, m22)
            z = 0

    elif order == "YXZ":
        x = math.asin(-clamp(m23, -1, 1))

        if abs(m23) < 0.9999999:
            y = math.atan2(m13, m33)
            z = math.atan2(m21, m22)
        else:
            y = math.atan2(-m31, m11)
            z = 0

    elif order == "ZXY":
        x = math.asin(clamp(m32, -1, 1))

        if abs(m32) < 0.9999999:
            y = math.atan2(-m31, m33)
            z = math.atan2(-m12, m22)
        else:
            y = 0
            z = math.atan2(m21, m11)

    elif order == "ZYX":
        y = math.asin(-clamp(m31, -1, 1))

        if abs(m31) < 0.9999999:
            x = math.atan2(m32, m33)
            z = math.atan2(m21, m11)
        else:
            x = 0
            z = math.atan2(-m12, m22)

    elif order == "YZX":
        z = math.asin(clamp(m21, -1, 1))

        if abs(m21) < 0.9999999:
            x = math.atan2(-m23, m22)
            y = math.atan2(-m31, m11)
        else:
            x = 0
            y = math.atan2(m13, m33)

    elif order == "XZY":
        z = math.asin(-clamp(m12, -1, 1))

        if abs(m12) < 0.9999999:
            x = math.atan2(m32, m22)
            y = math.atan2(m13, m11)
        else:
            x = math.atan2(-m23, m33)
            y = 0

    else:
        raise ValueError(f"Unknown order: {order}")

    return Euler(x, y, z)


def euler_from_quaternion(quaternion: Quaternion, order: str = "XYZ") -> Euler:
    matrix = matrix_rotation_from_quaternion(quaternion=quaternion)
    return euler_from_matrix(matrix, order)


def quaternion_from_euler(euler: Euler, order: str = "XYZ") -> Quaternion:
    x = euler.x
    y = euler.y
    z = euler.z

    c1 = math.cos(x / 2)
    c2 = math.cos(y / 2)
    c3 = math.cos(z / 2)

    s1 = math.sin(x / 2)
    s2 = math.sin(y / 2)
    s3 = math.sin(z / 2)

    quaternion = Quaternion()

    if order == "XYZ":
        quaternion.x = s1 * c2 * c3 + c1 * s2 * s3
        quaternion.y = c1 * s2 * c3 - s1 * c2 * s3
        quaternion.z = c1 * c2 * s3 + s1 * s2 * c3
        quaternion.w = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "YXZ":
        quaternion.x = s1 * c2 * c3 + c1 * s2 * s3
        quaternion.y = c1 * s2 * c3 - s1 * c2 * s3
        quaternion.z = c1 * c2 * s3 - s1 * s2 * c3
        quaternion.w = c1 * c2 * c3 + s1 * s2 * s3
    elif order == "ZXY":
        quaternion.x = s1 * c2 * c3 - c1 * s2 * s3
        quaternion.y = c1 * s2 * c3 + s1 * c2 * s3
        quaternion.z = c1 * c2 * s3 + s1 * s2 * c3
        quaternion.w = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "ZYX":
        quaternion.x = s1 * c2 * c3 - c1 * s2 * s3
        quaternion.y = c1 * s2 * c3 + s1 * c2 * s3
        quaternion.z = c1 * c2 * s3 - s1 * s2 * c3
        quaternion.w = c1 * c2 * c3 + s1 * s2 * s3
    elif order == "YZX":
        quaternion.x = s1 * c2 * c3 + c1 * s2 * s3
        quaternion.y = c1 * s2 * c3 + s1 * c2 * s3
        quaternion.z = c1 * c2 * s3 - s1 * s2 * c3
        quaternion.w = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "XZY":
        quaternion.x = s1 * c2 * c3 - c1 * s2 * s3
        quaternion.y = c1 * s2 * c3 - s1 * c2 * s3
        quaternion.z = c1 * c2 * s3 + s1 * s2 * c3
        quaternion.w = c1 * c2 * c3 + s1 * s2 * s3
    else:
        raise ValueError(f"Unknown order: {order}")

    quaternion.normalize()

    return quaternion


def vector_apply_quaternion(v: Vector3, q: Quaternion) -> Vector3:
    vx = v.x
    vy = v.y
    vz = v.z
    qx = q.x
    qy = q.y
    qz = q.z
    qw = q.w

    tx = 2 * (qy * vz - qz * vy)
    ty = 2 * (qz * vx - qx * vz)
    tz = 2 * (qx * vy - qy * vx)

    v.x = vx + qw * tx + qy * tz - qz * ty
    v.y = vy + qw * ty + qz * tx - qx * tz
    v.z = vz + qw * tz + qx * ty - qy * tx

    return v


def vectors_distance(v1: Vector3, v2: Vector3) -> float:
    dx = v1.x - v2.x
    dy = v1.y - v2.y
    dz = v1.z - v2.z

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def vector_apply_euler(v: Vector3, euler: Euler) -> Vector3:
    quaternion = quaternion_from_euler(euler)
    return vector_apply_quaternion(v, quaternion)


if __name__ == "__main__":

    v1 = vector_apply_euler(Vector3(-1, 0, 0), Euler(0, 0, 0))
    v2 = vector_apply_euler(Vector3(1, 0, 0), Euler(0, 0, 0))

    print(v1)
    print(v2)

    distance = vectors_distance(v1, v2)

    print(distance)
