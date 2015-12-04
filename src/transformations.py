#!/usr/bin/env python

import rospy
from geometry_msgs.msg import *
from std_msgs.msg import *
from numpy import ndarray, dot, sqrt, array, arccos, inner
import tf

__all__ = ['pose_to_list', 'quat_to_list', 'list_to_quat', 'list_to_pose', 'list_to_pose', 'quat_rotate', 'list_to_m4x4',
           'multiply_transform', 'scale_transform', 'inverse_transform', 'raw_list_to_list', 'distance', 'norm']

"""
This module extends a bit the tf module.
We should take some time one day to clean this and merge it into the official tf.transformations module
"""

def list_to_m4x4(pose_list):
    pos = tf.transformations.translation_matrix(pose_list[0])
    quat = tf.transformations.quaternion_matrix(pose_list[1])
    return dot(pos, quat)

def _is_indexable(var):
    try:
        var[0]
    except IndexError:
        return False
    return True

def pose_to_list(pose):
    """
    Convert a Pose or PoseStamped in Python list ((position), (quaternion))
    :param pose: geometry_msgs.msg.PoseStamped or geometry_msgs.msg.Pose
    :return: the equivalent in list ((position), (quaternion))
    """
    if type(pose)==geometry_msgs.msg.PoseStamped:
        return ((pose.pose.position.x, pose.pose.position.y, pose.pose.position.z),
                (pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w))
    elif type(pose)==geometry_msgs.msg.Pose:
        return ((pose.position.x, pose.position.y, pose.position.z),
                (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
    else:
        raise Exception("pose_to_list: parameter of type %s unexpected", str(type(pose)))

def list_to_quat(quatlist):
    """
    Convert a quaternion in the form of a list in geometry_msgs/Quaternion
    :param quatlist: [x, y, z, w]
    :return:
    """
    return geometry_msgs.msg.Quaternion(x= quatlist[0], y= quatlist[1], z= quatlist[2], w= quatlist[3])

def list_to_pose(poselist, frame_id="", stamp=rospy.Time(0)):
    """
    Convert a pose in the form of a list in PoseStamped
    :param poselist: a pose on the form [[x, y, z], [x, y, z, w]]
    :param frame_id: the frame_id on the outputed pose (facultative, empty otherwise)
    :param stamp: the stamp of the outputed pose (facultative, 0 otherwise)
    :return: the converted geometry_msgs/PoseStampted object
    """
    p = PoseStamped()
    p.header.frame_id = frame_id
    p.header.stamp = stamp
    p.pose.position.x = poselist[0][0]
    p.pose.position.y = poselist[0][1]
    p.pose.position.z = poselist[0][2]
    p.pose.orientation.x = poselist[1][0]
    p.pose.orientation.y = poselist[1][1]
    p.pose.orientation.z = poselist[1][2]
    p.pose.orientation.w = poselist[1][3]
    return p

def quat_to_list(quat):
    """
    convert a geometry_msgs/Quaternion or numpy.ndarray in list [x, y, z, w]
    :param quat:  a geometry_msgs/Quaternion or numpy.ndarray
    :return: the corresponding quaternion [x, y, z, w]
    """
    if isinstance(quat, Quaternion):
        return [quat.x, quat.y, quat.z, quat.w]
    elif isinstance(quat, ndarray) :
        return [quat[0], quat[1], quat[2], quat[3]]
    else:
        raise Exception("quat_to_list expects Quaternion only but received {}".format(str(type(quat))))


def quat_rotate(rotation, vector):
    """
    Rotate a vector according to a quaternion. Equivalent to the C++ method tf::quatRotate
    :param rotation: the rotation
    :param vector: the vector to rotate
    :return: the rotated vector
    """
    def quat_mult_quat(q1, q2):
        return (q1[0]*q2[0], q1[1]*q2[1], q1[2]*q2[2], q1[3]*q2[3])

    def quat_mult_point(q, w):
        return (q[3] * w[0] + q[1] * w[2] - q[2] * w[1],
                q[3] * w[1] + q[2] * w[0] - q[0] * w[2],
                q[3] * w[2] + q[0] * w[1] - q[1] * w[0],
                -q[0] * w[0] - q[1] * w[1] - q[2] * w[2])

    q = quat_mult_point(rotation, vector)
    q = tf.transformations.quaternion_multiply(q, tf.transformations.quaternion_inverse(rotation))
    return (q[0], q[1], q[2])

def multiply_transform(t1, t2):
    """
    Combines two transformations together
    The order is translation first, rotation then
    :param t1: [[x, y, z], [x, y, z, w]] or matrix 4x4
    :param t2: [[x, y, z], [x, y, z, w]] or matrix 4x4
    :return: The combination t1-t2 in the form [[x, y, z], [x, y, z, w]] or matrix 4x4
    """
    if _is_indexable(t1) and _is_indexable(t1[0]):
        t1m = tf.transformations.translation_matrix(t1[0])
        r1m = tf.transformations.quaternion_matrix(t1[1])
        m1m = dot(t1m, r1m)
        t2m = tf.transformations.translation_matrix(t2[0])
        r2m = tf.transformations.quaternion_matrix(t2[1])
        m2m = dot(t2m, r2m)
        rm = dot(m1m, m2m)
        rt = tf.transformations.translation_from_matrix(rm)
        rr = tf.transformations.quaternion_from_matrix(rm)
        return rt, rr
    else:
        return dot(t1, t2)

def scale_transform(t, scale):
    """
    Apply a scale to a transform
    :param t: The transform [[x, y, z], [x, y, z, w]] to be scaled OR the position [x, y, z] to be scaled
    :param scale: the scale of type float OR the scale [scale_x, scale_y, scale_z]
    :return: The scaled
    """
    if isinstance(scale, float) or isinstance(scale, int):
        scale = [scale, scale, scale]

    if _is_indexable(t[0]):
        return [[t[0][0]*scale[0], t[0][1]*scale[1], t[0][2]*scale[2]], t[1]]
    else:
        return [t[0]*scale[0], t[1]*scale[1], t[2]*scale[2]]

def inverse_transform(t):
    """
    Return the inverse transformation of t
    :param t: A transform [[x, y, z], [x, y, z, w]]
    :return: t2 such as multiply_transform_(t, t2) = [[0, 0, 0], [0, 0, 0, 1]]
    """
    return [quat_rotate(tf.transformations.quaternion_inverse(t[1]), [-t[0][0], -t[0][1], -t[0][2]]), tf.transformations.quaternion_inverse(t[1])]

def raw_list_to_list(t):
    """

    :param t: a raw list or tuple [x, y, z, x, y, z, w]
    :return: a formatted list [[x,y,z], [x,y,z,w]]
    """
    return [[t[0], t[1], t[2]],[t[3], t[4], t[5], t[6]]]

def distance(p1, p2):
    """
    Cartesian distance between two PoseStamped or PoseLists
    :param p1: point 1 (list or PoseStamped)
    :param p2: point 2 (list or PoseStamped)
    :return: cartesian distance (float)
    """
    if isinstance(p1, PoseStamped):
        x = p1.pose.position.x - p2.pose.position.x
        y = p1.pose.position.y - p2.pose.position.y
        z = p1.pose.position.z - p2.pose.position.z
    elif _is_indexable(p1[0]):
        x = p1[0][0] - p2[0][0]
        y = p1[0][1] - p2[0][1]
        z = p1[0][2] - p2[0][2]
    else:
        x = p1[0] - p2[0]
        y = p1[1] - p2[1]
        z = p1[2] - p2[2]
    return sqrt(x*x + y*y + z*z)

def distance_quat(p1, p2):
    """
    Returns the angle between two quaternions
    http://math.stackexchange.com/a/90098
    :param q1, q2: two quaternions [x, y, z, w] or two poses [[x, y, z], [x, y, z, w]]
    :return: the angle between them (radians)
    """
    if isinstance(p1, PoseStamped):
        p1 = pose_to_list(p1)
    if isinstance(p2, PoseStamped):
        p2 = pose_to_list(p2)
    if _is_indexable(p1):
        p1 = p1[1]
        p2 = p2[1]
    dotp = inner(array(p1), array(p2))
    return arccos(2*dotp*dotp-1)

def norm(point):
    """
    Norm between two PoseStamped or PoseLists
    :param point: point to compute the norm for (list or PoseStamped)
    :return: its norm (float)
    """
    if type(point)==PoseStamped:
        x = point.pose.position.x
        y = point.pose.position.y
        z = point.pose.position.z
    else:
        x = point[0][0]
        y = point[0][1]
        z = point[0][2]
    return sqrt(x*x + y*y + z*z)
