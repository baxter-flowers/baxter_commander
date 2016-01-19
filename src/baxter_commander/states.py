import tf
import copy


def state_to_pose_rpy(state, kinematic, arm):
    """
    Convert a robot state to a tuple of position and
    rpy rotation
    :param state: the RobotState
    :param arm: right or left, the robot arm
    :param kinematic: baxter_pykdl attribute
    :return: tuple of position and rpy rotation
    """
    def joints_to_dict(joint_values):
        """
        Create a dictionnary from the the joint values with
        joint names as keys
        :param joint_values: list of joint values
        :return: the dict of joint state
        """
        dict_joint = {}
        dict_joint[arm+'_s0'] = joint_values[0]
        dict_joint[arm+'_s1'] = joint_values[1]
        dict_joint[arm+'_e0'] = joint_values[2]
        dict_joint[arm+'_e1'] = joint_values[3]
        dict_joint[arm+'_w0'] = joint_values[4]
        dict_joint[arm+'_w1'] = joint_values[5]
        dict_joint[arm+'_w2'] = joint_values[6]
        return dict_joint
    dict_state = joints_to_dict(state.joint_state.position)
    fk = kinematic.forward_position_kinematics(dict_state)
    return fk[:3], tf.transformations.euler_from_quaternion(fk[-4:])


def correct_state_orientation(state, rpy, kinematic, arm):
    """
    Constrain a robot state to respect a specific orientation
    :param state: the RobotState
    :param rpy: the orientation to respect
    :param kinematic: baxter_pykdl attribute
    :return: new RobotState with orientation constrained
    """
    state = copy.deepcopy(state)
    pos, rot = state_to_pose_rpy(state, kinematic, arm)
    rpy_new = []
    for i in range(3):
        if rpy[i]:
            rpy_new.append(rpy[i])
        else:
            rpy_new.append(rot[i])
    quat_corrected = tuple(tf.transformations.quaternion_from_euler(rpy_new[0],
                                                                    rpy_new[1],
                                                                    rpy_new[2]))
    ik = kinematic.inverse_kinematics(pos, quat_corrected,
                                      seed=state.joint_state.position)
    joints_corrected = [ik[()][i] for i in range(7)]
    state_corrected = state
    state_corrected.joint_state.position = joints_corrected
    return state_corrected
