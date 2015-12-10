from nav_msgs.msg import Path
from moveit_msgs.msg import RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
import rospy



def states_to_trajectory(states, time_step=0.1):
    """
    Converts a list of RobotState/JointState in Robot Trajectory
    :param robot_states: list of RobotState
    :param time_step: duration in seconds between two consecutive points
    :return: a Robot Trajectory
    """
    rt = RobotTrajectory()
    for state_idx, state in enumerate(states):
        if isinstance(state, RobotState):
            state = state.joint_state

        jtp = JointTrajectoryPoint()
        jtp.positions = state.position
        #jtp.velocities = state.velocity  # Probably does not make sense to keep velocities and efforts here
        #jtp.effort = state.effort
        jtp.time_from_start = rospy.Duration(state_idx*time_step)
        rt.joint_trajectory.points.append(jtp)

    if len(states) > 0:
        if isinstance(states[0], RobotState):
            rt.joint_trajectory.joint_names = states[0].joint_state.name
        else:
            rt.joint_trajectory.joint_names = states[0].joint_names

    return rt