import tf
import copy


def state_to_pose(state, commander, rpy=False):
    """
    Convert a robot state to a tuple of position and
    rpy rotation
    :param state: the RobotState
    :param rpy: format for the orientation
    (True for rpy, False for quaternion)
    :return: tuple of position and rpy rotation
    """
    fk = commander.get_fk(robot_state=state)
    pos = fk[0]
    if rpy:
        rot = tf.transformations.euler_from_quaternion(fk[1])
    else:
        rot = fk[1]
    return pos, rot


def correct_state_orientation(state, rpy, commander):
    """
    Constrain a robot state to respect a specific orientation
    :param state: the RobotState
    :param rpy: the orientation to respect
    :param kinematic: baxter_pykdl attribute
    :return: new RobotState with orientation constrained
    """
    state = copy.deepcopy(state)
    pos, rot = state_to_pose(state, commander, True)
    rpy_new = []
    for i in range(3):
        if rpy[i]:
            rpy_new.append(rpy[i])
        else:
            rpy_new.append(rot[i])
    quat_corrected = tf.transformations.quaternion_from_euler(rpy_new[0],
                                                              rpy_new[1],
                                                              rpy_new[2])
    ik = commander.get_ik([pos.tolist(), quat_corrected], state)
    if ik is not None:
        return ik
    return state
