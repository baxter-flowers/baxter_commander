from nav_msgs.msg import Path
from moveit_msgs.msg import RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
import rospy
import numpy as np
from . import states
from copy import deepcopy
import matplotlib.pyplot as plt
import tf


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
    goal_state = [goal.joint_state.position[goal.joint_state.name.index(joint)] for joint in start.joint_state.name]

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

def generate_cartesian_path(path, frame_id, time, commander, start_state=None, n_points=50, max_speed=np.pi/4):
        """
        Generate a cartesian path of the end effector of "descent" meters where path = [x, y, z] wrt frame_id
        move_group.compute_cartesian_path does not allow to start from anything else but the current state, use this instead
        :param start_state: The start state to compute the trajectory from
        :param path: [x, y, z] The vector to follow in straight line
        :param n_points: The number of way points (high number will be longer to compute)
        :param time: time of the overall motion
        :param commander: Commander providing FK, IK and current state
        :param max_speed: Maximum speed in rad/sec for each joint
        :return: [rt, success_rate] a RobotTrajectory from the current pose and applying the given cartesian path to the
        end effector and the success rate of the motion (-1 in case of total failure)
        """
        pose_eef_approach = commander.get_fk(frame_id, start_state)
        waypoints = []
        for num in range(n_points):
            p = deepcopy(pose_eef_approach)
            p.pose.position.x += float(path[0]*num)/n_points
            p.pose.position.y += float(path[1]*num)/n_points
            p.pose.position.z += float(path[2]*num)/n_points
            waypoints.append(p)
        path = commander.get_ik(waypoints, start_state if start_state else commander.get_current_state()) # Provide the state here avoid the 1st point to jump

        if path[0] is None:
            return None, -1

        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = path[0].joint_state.name

        # "Jumps" detection: points with speed > max_speed are eliminated
        old_joint_state = None  # Will store the joint_state of the previous point...
        old_time_sec = float('inf')        # ...to compute the speed between 2 points in joint space
        jump_occurs = False

        for num, state in enumerate(path):
            time_sec = float(num*time)/len(path)
            if state:
                if old_time_sec < time_sec:
                    distance = (np.abs(old_joint_state-state.joint_state.position)%(np.pi))/(time_sec-old_time_sec)
                    jump_occurs = np.any(distance>max_speed)
                if not jump_occurs:
                    jtp = JointTrajectoryPoint()
                    jtp.positions = state.joint_state.position
                    jtp.time_from_start = rospy.Duration(time_sec)
                    trajectory.joint_trajectory.points.append(jtp)
                    old_joint_state = np.array(state.joint_state.position)
                    old_time_sec = time_sec

        successrate = float(len(trajectory.joint_trajectory.points))/n_points
        return trajectory, successrate

def generate_reverse_trajectory(trajectory):
    """
    Reverse the trajectory such as: state S -> trajectory T -> state B -> reverse_trajectory(T) -> state A
    :param trajectory: a RobotTrajectory
    :return: a RobotTrajectory
    """
    reversed = deepcopy(trajectory)
    reversed.joint_trajectory.points.reverse()
    n_points = len(trajectory.joint_trajectory.points)
    for p in range(n_points):
        reversed.joint_trajectory.points[p].time_from_start = trajectory.joint_trajectory.points[p].time_from_start
    return reversed

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


def get_position_array(trajectory):
    """
    Extract a 7-DoF position array from this RobotTrajectory
    :param trajectory: RobotTrajectory
    :return: position[joint][t]
    """
    return np.array([point.positions for point in trajectory.joint_trajectory.points]).T


def get_velocity_array(trajectory):
    """
    Extract a 7-DoF velocity array from this RobotTrajectory
    :param trajectory: RobotTrajectory
    :return: velocity[joint][t] or [] if velocity is not filled in
    """
    return np.array([point.velocities for point in trajectory.joint_trajectory.points]).T


def compute_velocity_array(trajectory):
    """
    Compute the velocity in rad/s from positions, the first point has a velocity of 0 rad/s
    :param trajectory: RobotTrajectory
    :return: velocity[joint][t]
    """
    points = []
    position = get_position_array(trajectory)
    time = get_time_array(trajectory)
    for joint in range(7):
        velocities = [0.]
        for point in range(1, len(time)):
            delta_pos = position[joint][point] - position[joint][point-1]
            delta_t = time[point] - time[point-1]
            velocities.append(delta_pos/delta_t)
        points.append(velocities)
    return np.array(points)


def get_time_array(trajectory):
    """
    Return the array of durations  (in sec) from the start of trajectory
    :param trajectory: RobotTrajectory
    :return: array of time from start e.g. [0.05, 1.5, 2.1, ...]
    """
    return np.array([point.time_from_start.to_sec() for point in trajectory.joint_trajectory.points])


def plot(trajectory, recompute_velocity=False):
    """
    Plot positions and velocities of all 7 joints
    :param trajectory: RobotTrajectory
    :param recompute_velocity: Compute the velocity by derivate the position instead of using provided data
    """
    position = get_position_array(trajectory)
    velocity = compute_velocity_array(trajectory) if recompute_velocity else get_velocity_array(trajectory)
    time = get_time_array(trajectory)
    colors = ['b', 'g', 'r', 'c', 'y', 'm', 'b']
    for joint_idx, joint in enumerate(['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2']):
        plt.plot(time, position[joint_idx], label=joint + ' (rad)', color=colors[joint_idx])
        if len(velocity) > 0:
            plt.plot(time, velocity[joint_idx], label=joint + ' (rad/s)', linestyle='--', color=colors[joint_idx])


def show():
    """
    Show the plot
    """
    plt.legend()
    plt.show()
