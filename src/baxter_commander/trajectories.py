from nav_msgs.msg import Path
from moveit_msgs.msg import RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
import rospy
import numpy as np


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
        # put name of joints to be moved
        rt.joint_trajectory.joint_names = start.joint_state.name
        return rt
