from nav_msgs.msg import Path
from moveit_msgs.msg import RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
import rospy
import numpy as np
from . import states
import tf
import copy


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
        # jtp.velocities = state.velocity
        # Probably does not make sense to keep velocities and efforts here
        # jtp.effort = state.effort
        jtp.time_from_start = rospy.Duration(state_idx*time_step)
        rt.joint_trajectory.points.append(jtp)

    if len(states) > 0:
        if isinstance(states[0], RobotState):
            rt.joint_trajectory.joint_names = states[0].joint_state.name
        else:
            rt.joint_trajectory.joint_names = states[0].joint_names

    return rt


def trapezoidal_speed_trajectory(goal, start,
                                 kv_max, ka_max,
                                 nb_points=100):
    """
    Calculate a trajectory from a start state (or current state)
    to a goal in joint space using a trapezoidal velocity model
    If no kv and ka max are given the default are used
    :param goal: A RobotState to be used as the goal of the trajectory
    :param nb_points: Number of joint-space points in the final trajectory
    :param kv_max: max K for velocity,
        can be a dictionary joint_name:value or a single value
    :param ka_max: max K for acceleration,
        can be a dictionary joint_name:value or a single value
    :param start: A RobotState to be used as the start state,
        joint order must be the same as the goal
    :return: The corresponding RobotTrajectory
    """
    def calculate_coeff(k, dist):
        coeff = []
        for i in range(len(dist)):
            min_value = 1
            for j in range(len(dist)):
                if i != j:
                    if k[i]*dist[j] > 0.0001:
                        min_value = min(min_value,
                                        (k[j]*dist[i]) / (k[i]*dist[j]))
            coeff.append(min_value)
        return coeff

    def calculate_max_speed(kv_des, ka, dist):
        kv = []
        for i in range(len(dist)):
            if dist[i] <= 1.5*kv_des[i]*kv_des[i]/ka[i]:
                kv.append(np.sqrt((2.0/3)*dist[i]*ka[i]))
            else:
                kv.append(kv_des[i])
        return kv

    def calculate_tau(kv, ka, lambda_i, mu_i):
        tau = []
        for i in range(len(kv)):
            if mu_i[i]*ka[i] > 0.0001:
                tau.append((3.0/2)*(lambda_i[i]*kv[i])/(mu_i[i]*ka[i]))
            else:
                tau.append(0.0)
        return tau

    def calculate_time(tau, lambda_i, kv, dist):
        time = []
        for i in range(len(tau)):
            if kv[i] > 0.0001:
                time.append(tau[i]+dist[i]/(lambda_i[i]*kv[i]))
            else:
                time.append(0.0)
        return time

    def calculate_joint_values(qi, D, tau, tf, nb_points):
        if tf > 0.0001:
            q_values = []
            time = np.linspace(0, tf, nb_points)
            for t in time:
                if t <= tau:
                    q_values.append(qi+D*(1.0/(2*(tf-tau))) *
                                    (2*t**3/(tau**2)-t**4/(tau**3)))
                elif t <= tf-tau:
                    q_values.append(qi+D*((2*t-tau)/(2*(tf-tau))))
                else:
                    q_values.append(qi+D*(1-(tf-t)**3/(2*(tf-tau)) *
                                    ((2*tau-tf+t)/(tau**3))))
        else:
            q_values = np.ones(nb_points)*qi
        return q_values

    # create the joint trajectory message
    rt = RobotTrajectory()
    joints = []
    start_state = start.joint_state.position
    goal_state = goal.joint_state.position

    # calculate the max joint velocity
    dist = np.array(goal_state) - np.array(start_state)
    abs_dist = np.absolute(dist)
    if isinstance(ka_max, dict):
        ka = np.ones(len(goal_state))*map(lambda name: ka_max[name],
                                          goal.joint_state.name)
    else:
        ka = np.ones(len(goal_state))*ka_max
    if isinstance(kv_max, dict):
        kv = np.ones(len(goal_state))*map(lambda name: kv_max[name],
                                          goal.joint_state.name)
    else:
        kv = np.ones(len(goal_state))*kv_max
    kv = calculate_max_speed(kv, ka, abs_dist)

    # calculate the synchronisation coefficients
    lambda_i = calculate_coeff(kv, abs_dist)
    mu_i = calculate_coeff(ka, abs_dist)

    # calculate the total time
    tau = calculate_tau(kv, ka, lambda_i, mu_i)
    tf = calculate_time(tau, lambda_i, kv, abs_dist)
    dt = np.array(tf).max()*(1.0/nb_points)

    if np.array(tf).max() > 0.0001:
        # calculate the joint value
        for j in range(len(goal_state)):
            pose_lin = calculate_joint_values(start_state[j], dist[j],
                                              tau[j], tf[j], nb_points+1)
            joints.append(pose_lin[1:])
        for i in range(nb_points):
            point = JointTrajectoryPoint()
            for j in range(len(goal_state)):
                point.positions.append(joints[j][i])
            # append the time from start of the position
            point.time_from_start = rospy.Duration.from_sec((i+1)*dt)
            # append the position to the message
            rt.joint_trajectory.points.append(point)
    else:
        point = JointTrajectoryPoint()
        point.positions = start_state
        point.time_from_start = rospy.Duration.from_sec(0)
        # append the position to the message
        rt.joint_trajectory.points.append(point)
    # put name of joints to be moved
    rt.joint_trajectory.joint_names = start.joint_state.name
    return rt


def minimum_jerk_trajectory(goal, start, commander, max_speed=0.2, nb_points=100, duration=None):
    """
    Calculate a trajectory from a start state (or current state)
    to a goal in task space using a minimum jerk model
    If no duration is given it is calculated based on
    the max_speed and the distance between start and goal
    :param goal: A RobotState to be used as the goal of the trajectory
    :param start: A RobotState to be used as the start state,
    :param nb_points: Number of joint-space points in the final trajectory
    :param max_speed: max task space seep
    :param duration: time of the motion
    :return: The corresponding RobotTrajectory
    """
    def dist_angle(a, b):
        return np.minimum(b-a, 2*np.pi-b+a)

    def compute(t):
        T = t/duration
        # compute the new state
        Xt = X0 + D*(10*T**3 - 15*T**4 + 6*T**5)
        return Xt

    # compute the forward kinematic for goal and start
    X0, rot0 = states.state_to_pose(start, commander, True)
    X0 = np.concatenate((X0, rot0))
    Xf, rotf = states.state_to_pose(goal, commander, True)
    Xf = np.concatenate((Xf, rotf))
    D_pos = Xf[:3] - X0[:3]
    D_rot = dist_angle(Xf[-3:], X0[-3:])
    D = np.concatenate((D_pos, D_rot))
    # calculate the time of the trajectory
    dist = np.linalg.norm(D)
    min_duration = dist/max_speed
    if duration is None:
        duration = min_duration
    else:
        if duration < min_duration:
            print "WARNING: Duration incoherent with maximum velocity. Time has been extended."
            duration = min_duration
    # create the time vector
    time_vect = np.linspace(0, duration, nb_points)
    # create the joint trajectory message
    rt = RobotTrajectory()
    point = JointTrajectoryPoint()
    point.positions = start.joint_state.position
    point.time_from_start = rospy.Duration.from_sec(0)
    # append the position to the message
    rt.joint_trajectory.points.append(point)
    # create intermediary_state for seed the ik
    inter = RobotState()
    inter.joint_state.name = start.joint_state.name
    # calculate the joint trajectory
    success = 0.
    if dist > 0.0001:
        for t in range(1, len(time_vect)):
            point = JointTrajectoryPoint()
            # calculate task space point
            Xt = compute(time_vect[t])
            # convert it in pos/quaternion
            pos = Xt[:3]
            rot = tf.transformations.quaternion_from_euler(Xt[-3], Xt[-2], Xt[-1])
            # collect the previous state
            inter.joint_state.position = rt.joint_trajectory.points[-1].positions
            # find the inverse kinematic
            ik = commander.get_ik([pos.tolist(), rot], inter)
            if ik is not None:
                # append the position to the message
                point.positions = ik.joint_state.position
                point.time_from_start = rospy.Duration.from_sec(time_vect[t])
                rt.joint_trajectory.points.append(point)
                success += 1
    # put name of joints to be moved
    rt.joint_trajectory.joint_names = start.joint_state.name
    success /= nb_points
    return rt, success
