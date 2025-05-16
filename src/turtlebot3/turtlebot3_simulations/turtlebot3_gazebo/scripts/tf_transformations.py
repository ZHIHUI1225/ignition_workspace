#!/usr/bin/env python3

# This file provides compatibility functions for tf_transformations
# It implements the most commonly used functions from the ROS1 tf package
# for quaternion and euler angle conversions

import numpy as np
import math

def quaternion_from_euler(ai, aj, ak):
    """
    Return a quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    Default rotation sequence is 'xyz' (roll, pitch, yaw)
    """
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = np.empty((4, ))
    q[0] = cj*cc + sj*ss  # w
    q[1] = cj*sc - sj*cs  # x
    q[2] = cj*ss + sj*cc  # y
    q[3] = cj*cs - sj*sc  # z

    return q

def euler_from_quaternion(q):
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    q : Quaternion [x, y, z, w]

    Return: Roll, pitch, yaw angles as an array
    """
    x, y, z, w = q

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(sinp) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.

    q1, q2 : Quaternions [x, y, z, w]

    Return: q1 * q2 as an array
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    q = np.empty((4, ))
    q[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # w
    q[1] = x1 * w2 + w1 * x2 - z1 * y2 + y1 * z2  # x
    q[2] = y1 * w2 + z1 * x2 + w1 * y2 - x1 * z2  # y
    q[3] = z1 * w2 - y1 * x2 + x1 * y2 + w1 * z2  # z

    return q

def quaternion_inverse(q):
    """
    Return the inverse of a quaternion.

    q : Quaternion [x, y, z, w]

    Return: q^-1 (inverse quaternion) as an array
    """
    q = np.array(q)
    q_inv = np.empty((4, ))
    q_inv[0] = q[0]  # w
    q_inv[1:] = -q[1:]  # x, y, z

    return q_inv / np.sum(q * q)
