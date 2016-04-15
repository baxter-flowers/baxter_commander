import rospy
from threading import Lock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import RobotTrajectory
from nav_msgs.msg import Path
from baxter_core_msgs.msg import EndEffectorState

__all__ = ['Recorder']

class Recorder:
    def __init__(self, side='both', max_points=1000000):
        """
        Constructs a recorder in task space and joint space
        :param side: both, left or right
        :param max_points: Maximum number of trajectory points to be recorded
        """
        self._side = side
        self.max_n_points = max_points
        self._recording = False
        self._recorder_rate = -1

        self._joint_last_recorded_point = rospy.Time(0)
        self._joint_recorder_lock = Lock()
        self._joint_recorded_points = []
        self._joint_recorded_times = []

        self._eef_last_recorded_point = rospy.Time(0)
        self._eef_recorder_lock = Lock()
        self._eef_recorded_points = []
        self._eef_recorded_times = []

        sides = ['left', 'right', 'both']
        if side not in sides:
            raise ValueError("Side '{}' can't be recorded. Accepted values are {}".format(side, str(sides)))

        rospy.Subscriber("/robot/joint_states", JointState, self._cb_joint_state)
        rospy.Subscriber("/robot/limb/{}/endpoint_state".format(side), EndEffectorState, self._cb_eef_state)

    def _cb_joint_state(self, msg):
        if len(self._joint_recorded_points) < self.max_n_points:
            now = rospy.Time.now()
            if self._recording and self._joint_last_recorded_point + rospy.Duration(1./self._recorder_rate) < now:
                self._joint_last_recorded_point = now
                with self._joint_recorder_lock:
                    self._joint_recorded_points.append(msg)
                    self._joint_recorded_times.append(msg.header.stamp)

    def _cb_eef_state(self, msg):
        if len(self._eef_recorded_points) < self.max_n_points:
            now = rospy.Time.now()
            if self._recording and self._eef_last_recorded_point + rospy.Duration(1./self._recorder_rate) < now:
                self._eef_last_recorded_point = now
                with self._eef_recorder_lock:
                    self._eef_recorded_points.append(msg)
                    self._eef_recorded_times.append(msg.header.stamp)

    def start(self, rate_hz=50.):
        """
        Start the recording of end effector and joint points at the specified rate
        :param rate_hz: Recording rate
        """
        if self._recording:
            rospy.logerr("Recorder is already recording {} arm".format(str(self._side)))
            return
        self._recorder_rate = rate_hz
        with self._joint_recorder_lock:
            self._joint_recorded_points = []
            self._joint_recorded_times = []
        with self._eef_recorder_lock:
            self._eef_recorded_points = []
            self._eef_recorded_times = []
        self._recording = True

    def stop(self, include_velocity=False, include_effort=False):
        """
        Stop the recording and returns the recorded trajectory of joints declared in the constructor
        :param include_velocity:  if True, the joint velocities will be included in the returned trajectory
        :param include_effort:  if True, the joint efforts will be included in the returned trajectory
        :return: (joint_traj, eef_traj) resp. the recorded joint trajectory (RobotTrajectory) and end effector trajectory in frame 'base' (Path)
        """
        self._recording = False

        # Now we reconstruct a RobotTrajectory...
        rt = RobotTrajectory()
        rt.joint_trajectory.joint_names = [joint for joint in self._joint_recorded_points[0].name if self._side in joint or
                                           self._side == 'both'] if len(self._joint_recorded_points) > 0 else []
        with self._joint_recorder_lock:
            for idx, js in enumerate(self._joint_recorded_points):
                jtp = JointTrajectoryPoint()
                try:
                    jtp.positions = [js.position[js.name.index(joint)] for joint in rt.joint_trajectory.joint_names]
                except ValueError:
                    # This specific JS sample does not contain the desired joints, skip it
                    continue
                else:
                    if include_velocity:
                        jtp.velocities = [js.velocity[js.name.index(joint)] for joint in rt.joint_trajectory.joint_names]
                    if include_effort:
                        jtp.accelerations = [js.effort[js.name.index(joint)] for joint in rt.joint_trajectory.joint_names]
                    jtp.time_from_start = self._joint_recorded_times[idx] - self._joint_recorded_times[0]
                    rt.joint_trajectory.points.append(jtp)

        # ... as well as an End Effector trajectory
        eft = Path()
        with self._eef_recorder_lock:
            for eef in self._eef_recorded_points:
                ps = PoseStamped()
                ps.header = eef.header
                ps.header.frame_id = 'base'
                ps.pose = eef.pose
                eft.poses.append(ps)
                # twist and wrench in eef are discarded
        eft.header.frame_id = 'base'

        return rt, eft