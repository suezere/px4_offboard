import numpy as np
import transforms3d as t3d
from pyquaternion import Quaternion

def ned_to_enu(enu_position):
    rotation_matrix = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    ned_position = np.dot(rotation_matrix, enu_position)
    return ned_position

def ned_to_enu_quaternion(q):
    q_enu = Quaternion(q[0],q[2],q[1], - q[3])
    q_90 = Quaternion(axis = [0, 0, 1], degrees = 90)
    q_enu = q_enu *q_90
    return np.array([q_enu[0], q_enu[1], q_enu[2], q_enu[3]])

def Quaternion2Euler(quaternion):
    """
    converts a quaternion attitude to an euler angle attitude
    :param quaternion: the quaternion to be converted to euler angles in a np.matrix
    :return: the euler angle equivalent (phi, theta, psi) in a np.array
    """
    e0 = quaternion[0]
    e1 = quaternion[1]
    e2 = quaternion[2]
    e3 = quaternion[3]
    phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2.0 + e3**2.0 - e1**2.0 - e2**2.0)
    theta = np.arcsin(2.0 * (e0 * e2 - e1 * e3))
    psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2.0 + e1**2.0 - e2**2.0 - e3**2.0)

    return phi, theta, psi

def Euler2Quaternion(phi, theta, psi):
    """
    Converts an euler angle attitude to a quaternian attitude
    :param euler: Euler angle attitude in a np.matrix(phi, theta, psi)
    :return: Quaternian attitude in np.array(e0, e1, e2, e3)
    """

    e0 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)
    e1 = np.cos(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0) - np.sin(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0)
    e2 = np.cos(psi/2.0) * np.sin(theta/2.0) * np.cos(phi/2.0) + np.sin(psi/2.0) * np.cos(theta/2.0) * np.sin(phi/2.0)
    e3 = np.sin(psi/2.0) * np.cos(theta/2.0) * np.cos(phi/2.0) - np.cos(psi/2.0) * np.sin(theta/2.0) * np.sin(phi/2.0)

    return np.asfarray([e0, e1, e2, e3], dtype = np.float32)



