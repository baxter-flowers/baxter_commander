import numpy as np
import rospy
import xmltodict

from baxter_pykdl import baxter_kinematics
from baxter_interface import Limb, Gripper
from baxter_core_msgs.msg import CollisionDetectionState, DigitalIOState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest

from actionlib import SimpleGoalState, SimpleActionClient
from moveit_msgs.msg import RobotTrajectory, RobotState, DisplayTrajectory
from moveit_msgs.srv import GetPositionFKRequest, GetPositionFK, GetPositionIKRequest, GetPositionIK
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from threading import Lock
from copy import deepcopy
from transformations import pose_to_list, list_to_pose
from tf import TransformListener

from . recorder import Recorder
from . import trajectories
from . import states

__all__ = ['ArmCommander']

class ArmCommander(Limb):
    """
    This class overloads Limb from the  Baxter Python SDK adding the support of trajectories via RobotState and RobotTrajectory messages
    Allows to control the entire arm either in joint space, or in task space, or (later) with path planning, all with simulation
    """
    def __init__(self, name, rate=100, kinematics='robot', default_kv_max=1., default_ka_max=0.5):
        """
        :param name: 'left' or 'right'
        :param rate: Rate of the control loop for execution of motions
        :param kinematics: Kinematics solver, "robot", "kdl" or "ros"
        :param default_kv_max: Default K maximum for velocity
        :param default_ka_max: Default K maximum for acceleration
        """
        Limb.__init__(self, name)
        self._world = 'base'
        self.kv_max = default_kv_max
        self.ka_max = default_ka_max
        self._gripper = Gripper(name)
        self._rate = rospy.Rate(rate)
        self._tf_listener = TransformListener()
        self.recorder = Recorder(name)

        # Kinematics services: Selection among different services
        self._kinematics_selected = kinematics
        self._kinematics_services = {'kdl': {'fk': self._get_fk_pykdl, 'ik': self._get_ik_pykdl},
                                     'robot': {'fk': self._get_fk_robot, 'ik': self._get_ik_robot},
                                     'ros': {'fk': self._get_fk_ros, 'ik': self._get_ik_ros}}

        # Kinematics services: PyKDL
        self._kinematics_pykdl = baxter_kinematics(name)

        # Kinematics services: Robot
        #self._f_kinematics_robot_name = 'ExternalTools/{}/PositionKinematicsNode/FKService'.format(name)
        #self._f_kinematics_robot = rospy.ServiceProxy(self._f_kinematics_robot_name, ) # This service doesnt exist?
        self._i_kinematics_robot_name = 'ExternalTools/{}/PositionKinematicsNode/IKService'.format(name)
        self._i_kinematics_robot = rospy.ServiceProxy(self._i_kinematics_robot_name, SolvePositionIK)

        # Kinematics services: ROS
        self._f_kinematics_ros_name = '/compute_fk'
        self._f_kinematics_ros = rospy.ServiceProxy(self._f_kinematics_ros_name, GetPositionFK)
        self._i_kinematics_ros_name = '/compute_ik'
        self._i_kinematics_ros = rospy.ServiceProxy(self._i_kinematics_ros_name, GetPositionIK)

        # Execution attributes
        rospy.Subscriber('/robot/limb/{}/collision_detection_state'.format(name), CollisionDetectionState, self._cb_collision, queue_size=1)
        rospy.Subscriber('/robot/digital_io/{}_lower_cuff/state'.format(name), DigitalIOState, self._cb_dig_io, queue_size=1)
        self._stop_reason = ''  # 'cuff' or 'collision' could cause a trajectory to be stopped
        self._stop_lock = Lock()
        action_server_name = "/robot/limb/{}/follow_joint_trajectory".format(self.name)
        self.client = SimpleActionClient(action_server_name, FollowJointTrajectoryAction)

        self._display_traj = rospy.Publisher("/move_group/display_planned_path", DisplayTrajectory, queue_size=1)
        self._gripper.calibrate()

        rospy.loginfo("ArmCommander({}): Waiting for action server {}...".format(self.name, action_server_name))
        self.client.wait_for_server()

    ######################################### CALLBACKS #########################################
    def _cb_collision(self, msg):
        if msg.collision_state:
            with self._stop_lock:
                self._stop_reason = 'collision'

    def _cb_dig_io(self, msg):
        if msg.state > 0:
            with self._stop_lock:
                self._stop_reason = 'cuff'
    #############################################################################################

    def endpoint_pose(self):
        """
        Returns the pose of the end effector
        :return: [[x, y, z], [x, y, z, w]]
        """
        pose = Limb.endpoint_pose(self)
        return [[pose['position'].x, pose['position'].y, pose['position'].z],
                [pose['orientation'].x, pose['orientation'].y, pose['orientation'].z, pose['orientation'].w]]

    def endpoint_name(self):
        return self.name+'_gripper'

    def group_name(self):
        return self.name+'_arm'

    def joint_limits(self):
        xml_urdf = rospy.get_param('robot_description')
        dict_urdf = xmltodict.parse(xml_urdf)
        joints_urdf = []
        joints_urdf.append([j['@name'] for j in dict_urdf['robot']['joint'] if j['@name'] in self.joint_names()])
        joints_urdf.append([[float(j['limit']['@lower']), float(j['limit']['@upper'])] for j in dict_urdf['robot']['joint'] if j['@name'] in self.joint_names()])
        # reorder the joints limits
        return dict(zip(self.joint_names(),
                        [joints_urdf[1][joints_urdf[0].index(name)] for name in self.joint_names()]))

    def get_current_state(self, list_joint_names=[]):
        """
        Returns the current RobotState describing all joint states
        :param list_joint_names: If not empty, returns only the state of the requested joints
        :return: a RobotState corresponding to the current state read on /robot/joint_states
        """
        if len(list_joint_names) == 0:
            list_joint_names = self.joint_names()
        state = RobotState()
        state.joint_state.name = list_joint_names
        state.joint_state.position = map(self.joint_angle, list_joint_names)
        state.joint_state.velocity = map(self.joint_velocity, list_joint_names)
        state.joint_state.effort = map(self.joint_effort, list_joint_names)
        return state

    def get_ik(self, eef_poses, seeds=[]):
        """
        Return IK solutions of this arm's end effector according to the method declared in the constructor
        :param eef_poses: a PoseStamped or a list [[x, y, z], [x, y, z, w]] in world frame or a list of PoseStamped
        :param seeds: a single seed or a list of seeds of type RobotState for each input pose
        :return: a list of RobotState for each input pose in input or a single RobotState
        TODO: accept also a Point (baxter_pykdl's IK accepts orientation=None)
        Child methods wait for a *list* of pose(s) and a *list* of seed(s) for each pose
        """
        if not isinstance(eef_poses, list) or isinstance(eef_poses[0], list) and not isinstance(eef_poses[0][0], list):
            eef_poses = [eef_poses]

        if not seeds:
            seeds=[]
        elif not isinstance(seeds, list):
            seeds = [seeds]*len(eef_poses)

        input = []
        for eef_pose in eef_poses:
            if isinstance(eef_pose, list):
                input.append(list_to_pose(eef_pose, self._world))
            elif isinstance(eef_pose, PoseStamped):
                input.append(eef_pose)
            else:
                raise TypeError("ArmCommander.get_ik() accepts only a list of Postamped or [[x, y, z], [x, y, z, w]], got {}".format(str(type(eef_pose))))

        output = self._kinematics_services[self._kinematics_selected]['ik'](input, seeds)
        return output if len(eef_poses)>1 else output[0]

    def get_fk(self, frame_id=None, robot_state=None):
        """
        Return The FK solution oth this robot state according to the method declared in the constructor
        robot_state = None will give the current endpoint pose in frame_id
        :param robot_state: a RobotState message
        :return: [[x, y, z], [x, y, z, w]]
        """
        if frame_id is None:
            frame_id = self._world
        if isinstance(robot_state, RobotState) or robot_state is None:
            return self._kinematics_services[self._kinematics_selected]['fk'](frame_id, robot_state)
        else:
            raise TypeError("ArmCommander.get_fk() accepts only a RobotState, got {}".format(str(type(robot_state))))

    def _get_fk_pykdl(self, frame_id, state=None):
        if state is None:
            state = self.get_current_state()
        fk = self._kinematics_pykdl.forward_position_kinematics(dict(zip(state.joint_state.name, state.joint_state.position)))
        return [fk[:3], fk[-4:]]

    def _get_fk_robot(self, frame_id = None, state=None):
        if state is not None:
            raise NotImplementedError("_get_fk_robot has no FK service provided by the robot except for its current endpoint pose")
        ps = list_to_pose(self.endpoint_pose(), self._world)
        return self._tf_listener.transformPose(frame_id, ps)

    def _get_fk_ros(self, frame_id = None, state=None):
        rqst = GetPositionFKRequest()
        rqst.header.frame_id = self._world if frame_id is None else frame_id
        rqst.fk_link_names = [self.endpoint_name()]
        if isinstance(state, RobotState):
            rqst.robot_state = state
        elif isinstance(state, JointState):
            rqst.robot_state.joint_state = state
        elif state is None:
            rqst.robot_state = self.get_current_state()
        else:
            raise AttributeError("Provided state is an invalid type")
        rospy.wait_for_service(self._f_kinematics_ros_name)
        fk_answer = self._f_kinematics_ros.call(rqst)

        if fk_answer.error_code.val==1:
            return fk_answer.pose_stamped[0]
        else:
            return None

    def _get_ik_pykdl(self, eef_poses, seeds=[]):
        solutions = []
        for pose_num, eef_pose in enumerate(eef_poses):
            if eef_pose.header.frame_id.strip('/') != self._world.strip('/'):
                raise NotImplementedError("_get_ik_pykdl: Baxter PyKDL implementation does not accept frame_ids other than {}".format(self._world))

            pose = pose_to_list(eef_pose)
            resp = self._kinematics_pykdl.inverse_kinematics(pose[0], pose[1],
                                                             [seeds[pose_num].joint_state.position[seeds[pose_num].joint_state.name.index(joint)]
                                                              for joint in self.joint_names()] if len(seeds)>0 else None)
            if resp is None:
                rs = None
            else:
                rs = RobotState()
                rs.is_diff = False
                rs.joint_state.name = self.joint_names()
                rs.joint_state.position = resp
            solutions.append(rs)
        return solutions

    def _get_ik_robot(self, eef_poses, seeds=[]):
        ik_req = SolvePositionIKRequest()

        for eef_pose in eef_poses:
            ik_req.pose_stamp.append(eef_pose)

        ik_req.seed_mode = ik_req.SEED_USER if len(seeds) > 0 else ik_req.SEED_CURRENT
        for seed in seeds:
            ik_req.seed_angles.append(seed.joint_state)

        rospy.wait_for_service(self._i_kinematics_robot_name, 5.0)
        resp = self._i_kinematics_robot(ik_req)

        solutions = []
        for j, v in zip(resp.joints, resp.isValid):
            solutions.append(RobotState(is_diff=False, joint_state=j) if v else None)
        return solutions

    def _get_ik_ros(self, eef_poses, seeds=[]):
        rqst = GetPositionIKRequest()
        rqst.ik_request.avoid_collisions = True
        rqst.ik_request.group_name = self.group_name()
        rospy.wait_for_service(self._i_kinematics_ros_name)

        solutions = []
        for pose_num, eef_pose in enumerate(eef_poses):
            rqst.ik_request.pose_stamped = eef_pose  # Do we really to do a separate call for each pose? _vector not used
            ik_answer = self._i_kinematics_ros.call(rqst)

            if len(seeds) > 0:
                rqst.ik_request.robot_state = seeds[pose_num]

            if ik_answer.error_code.val==1:
                # Apply a filter to return only joints of this group
                try:
                    ik_answer.solution.joint_state.velocity = [value for id_joint, value in enumerate(ik_answer.solution.joint_state.velocity) if ik_answer.solution.joint_state.name[id_joint] in self.joint_names()]
                    ik_answer.solution.joint_state.effort = [value for id_joint, value in enumerate(ik_answer.solution.joint_state.effort) if ik_answer.solution.joint_state.name[id_joint] in self.joint_names()]
                except IndexError:
                    pass
                ik_answer.solution.joint_state.position = [value for id_joint, value in enumerate(ik_answer.solution.joint_state.position) if ik_answer.solution.joint_state.name[id_joint] in self.joint_names()]
                ik_answer.solution.joint_state.name = [joint for joint in ik_answer.solution.joint_state.name if joint in self.joint_names()]
                solutions.append(ik_answer.solution)
            else:
                solutions.append(None)
        return solutions

    def translate_to_cartesian(self, path, frame_id, time, n_points=50, max_speed=np.pi/4, min_success_rate=0.5, display_only=False,
                                     timeout=0, stop_test=lambda:False, pause_test=lambda:False):
        """
        Translate the end effector in straight line, following path=[translate_x, translate_y, translate_z] wrt frame_id
        :param path: Path [x, y, z] to follow wrt frame_id
        :param frame_id: frame_id of the given input path
        :param time: Time of the generated trajectory
        :param n_points: Number of 3D points of the cartesian trajectory
        :param max_speed: Maximum speed for every single joint in rad.s-1, allowing to avoid jumps in joints configuration
        :param min_success_rate: Raise RuntimeError in case the success rate is lower than min_success_rate
        :param display_only:
        :param timeout: In case of cuff interaction, indicates the max time to retry before giving up (default is 0 = do not retry)
        :param stop_test: pointer to a function that returns True if execution must stop now. /!\ Should be fast, it will be called at 100Hz!
        :param pause_test: pointer to a function that returns True if execution must pause now. If yes it blocks until pause=False again and relaunches the same goal
        /!\ Test functions must be fast, they will be called at 100Hz!
        :return:
        :raises: RuntimeError if success rate is too low
        """
        def trajectory_callable(start):
            traj, success_rate = trajectories.generate_cartesian_path(path, frame_id, time, self, None, n_points, max_speed)
            if success_rate < min_success_rate:
                raise RuntimeError("Unable to generate cartesian path (success rate : {})".format(success_rate))
            return traj
        return self._relaunched_move_to(trajectory_callable, display_only, timeout, stop_test, pause_test)

    def move_to_controlled(self, goal, rpy=[0, 0, 0], display_only=False, timeout=15, stop_test=lambda:False, pause_test=lambda:False):
        """
        Move to a goal using interpolation in joint space with limitation of velocity and acceleration
        :param goal: Pose, PoseStamped or RobotState
        :param rpy: Vector [Roll, Pitch, Yaw] filled with 0 if this axis must be preserved, 1 otherwise
        :param display_only:
        :param timeout: In case of cuff interaction, indicates the max time to retry before giving up
        :param stop_test: pointer to a function that returns True if execution must stop now. /!\ Should be fast, it will be called at 100Hz!
        :param pause_test: pointer to a function that returns True if execution must pause now. If yes it blocks until pause=False again and relaunches the same goal
        /!\ Test functions must be fast, they will be called at 100Hz!
        :return: None
        :raises: ValueError if IK has no solution
        """
        assert callable(stop_test)
        assert callable(pause_test)

        if not isinstance(goal, RobotState):
            goal = self.get_ik(goal)
        if goal is None:
            raise ValueError('This goal is not reachable')

        # collect the robot state
        start = self.get_current_state()

        # correct the orientation if rpy is set
        if np.array(rpy).any():
            # convert the starting point to rpy pose
            pos, rot = states.state_to_pose(start,
                                            self,
                                            True)
            for i in range(3):
                if rpy[i]:
                    rpy[i] = rot[i]
            goal = states.correct_state_orientation(goal, rpy, self)

        # parameters for trapezoidal method
        kv_max = self.kv_max
        ka_max = self.ka_max

        return self._relaunched_move_to(lambda start: trajectories.trapezoidal_speed_trajectory(goal, start=start ,kv_max=kv_max, ka_max=ka_max),
                                        display_only, timeout, stop_test, pause_test)

    def rotate_joint(self, joint_name, goal_angle, display_only=False, stop_test=lambda:False, pause_test=lambda:False):
        goal = self.get_current_state()
        joint = goal.joint_state.name.index(joint_name)
        # JTAS accepts all angles even out of limits
        #limits = self.joint_limits()[joint_name]
        goal.joint_state.position[joint] = goal_angle
        return self.move_to_controlled(goal, display_only=display_only, stop_test=stop_test, pause_test=pause_test)

    def _relaunched_move_to(self, trajectory_callable, display_only=False, timeout=15, stop_test=lambda:False, pause_test=lambda:False):
        """
        Relaunch several times (until cuff interaction or failure) a move_to() whose trajectory is generated by the callable passed in parameter
        :param trajectory_callable: Callable to call to recompute the trajectory
        :param display_only:
        :param timeout: In case of cuff interaction, indicates the max time to retry before giving up
        :param stop_test: pointer to a function that returns True if execution must stop now. /!\ Should be fast, it will be called at 100Hz!
        :param pause_test: pointer to a function that returns True if execution must pause now. If yes it blocks until pause=False again and relaunches the same goal
        :return:
        """
        assert callable(trajectory_callable)

        retry = True
        t0 = rospy.get_time()
        while retry and rospy.get_time()-t0 < timeout or timeout == 0:
            start = self.get_current_state()
            trajectory = trajectory_callable(start)

            if display_only:
                self.display(trajectory)
                break
            else:
                retry = not self.execute(trajectory, test=lambda: stop_test() or pause_test())
                if pause_test():
                    if not stop_test():
                        retry = True
                    while pause_test():
                        rospy.sleep(0.1)
            if timeout == 0:
                return not display_only and not retry
            if retry:
                rospy.sleep(1)
        return not display_only and not retry

    def get_random_pose(self):
        # get joint names
        joint_names = self.joint_names()
        # create a random joint state
        bounds = []
        for key, value in self.joint_limits().iteritems():
            bounds.append(value)
        bounds = np.array(bounds)
        joint_state = np.random.uniform(bounds[:, 0], bounds[:, 1], len(joint_names))
        # append it in a robot state
        goal = RobotState()
        goal.joint_state.name = joint_names
        goal.joint_state.position = joint_state
        goal.joint_state.header.stamp = rospy.Time.now()
        goal.joint_state.header.frame_id = 'base'
        return goal

    ######################## OPERATIONS ON TRAJECTORIES
    # TO BE MOVED IN trajectories.py
    def interpolate_joint_space(self, goal, total_time, nb_points, start=None):
        """
        Interpolate a trajectory from a start state (or current state) to a goal in joint space
        :param goal: A RobotState to be used as the goal of the trajectory
        param total_time: The time to execute the trajectory
        :param nb_points: Number of joint-space points in the final trajectory
        :param start: A RobotState to be used as the start state, joint order must be the same as the goal
        :return: The corresponding RobotTrajectory
        """
        dt = total_time*(1.0/nb_points)
        # create the joint trajectory message
        traj_msg = JointTrajectory()
        rt = RobotTrajectory()
        if start == None:
            start = self.get_current_state()
        joints = []
        start_state = start.joint_state.position
        goal_state = goal.joint_state.position
        for j in range(len(goal_state)):
            pose_lin = np.linspace(start_state[j],goal_state[j],nb_points+1)
            joints.append(pose_lin[1:].tolist())
        for i in range(nb_points):
            point = JointTrajectoryPoint()
            for j in range(len(goal_state)):
                point.positions.append(joints[j][i])
            # append the time from start of the position
            point.time_from_start = rospy.Duration.from_sec((i+1)*dt)
            # append the position to the message
            traj_msg.points.append(point)
        # put name of joints to be moved
        traj_msg.joint_names = self.joint_names()
        # send the message
        rt.joint_trajectory = traj_msg
        return rt

    def display(self, trajectory):
        """
        Display a joint-space trajectory in RVIz loaded with the Moveit plugin
        :param trajectory: a RobotTrajectory message
        """
        # Publish the DisplayTrajectory (only for trajectory preview in RViz)
        # includes a convert of float durations in rospy.Duration()
        dt = DisplayTrajectory()
        if isinstance(trajectory, RobotTrajectory):
            dt.trajectory.append(trajectory)
        elif isinstance(trajectory, JointTrajectory):
            rt = RobotTrajectory()
            rt.joint_trajectory = trajectory
            dt.trajectory.append(rt)
        else:
            raise NotImplementedError("ArmCommander.display() expected type RobotTrajectory or JointTrajectory, got {}".format(str(type(trajectory))))
        self._display_traj.publish(dt)

    def execute(self, trajectory, test=None):
        """
        Safely executes a trajectory in joint space on the robot or simulate through RViz and its Moveit plugin (File moveit.rviz must be loaded into RViz)
        This method is BLOCKING until the command succeeds or failure occurs i.e. the user interacted with the cuff or collision has been detected
        Non-blocking needs should deal with the JointTrajectory action server
        :param trajectory: either a RobotTrajectory or a JointTrajectory
        :param test: pointer to a function that returns True if execution must stop now. /!\ Should be fast, it will be called at 100Hz!
        :return: True if execution ended successfully, False otherwise
        """
        self.display(trajectory)
        with self._stop_lock:
            self._stop_reason = ''
        if isinstance(trajectory, RobotTrajectory):
            trajectory = trajectory.joint_trajectory
        elif isinstance(trajectory, JointTrajectory):
            trajectory = trajectory
        ftg = FollowJointTrajectoryGoal()
        ftg.trajectory = trajectory
        self.client.send_goal(ftg)
        # Blocking part, wait for the callback or a collision or a user manipulation to stop the trajectory

        while self.client.simple_state != SimpleGoalState.DONE:
            if callable(test) and test():
                self.client.cancel_goal()
                return True

            if self._stop_reason!='':
                self.client.cancel_goal()
                return False

            self._rate.sleep()

        return True

    def close(self):
        """
        Open the gripper
        :return: True if an object has been grasped
        """
        return self._gripper.close(True)

    def open(self):
        """
        Close the gripper
        return: True if an object has been released
        """
        return self._gripper.open(True)

    def gripping(self):
        return self._gripper.gripping()

    def wait_for_human_grasp(self, threshold=1, rate=10, ignore_gripping=True):
        """
        Blocks until external motion is given to the arm
        :param threshold:
        :param rate: rate of control loop in Hertz
        :param ignore_gripping: True if we must wait even if no object is gripped
        """
        def detect_variation():
            new_effort = np.array(self.get_current_state([self.name+'_w0',
                                                          self.name+'_w1',
                                                          self.name+'_w2']).joint_state.effort)
            delta = np.absolute(effort - new_effort)
            return np.amax(delta) > threshold
        # record the effort at calling time
        effort = np.array(self.get_current_state([self.name+'_w0',
                                                  self.name+'_w1',
                                                  self.name+'_w2']).joint_state.effort)
        # loop till the detection of changing effort
        rate = rospy.Rate(rate)
        while not detect_variation() and not rospy.is_shutdown() and (ignore_gripping or self.gripping()):
            rate.sleep()