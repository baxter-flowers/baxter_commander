import rospy
from threading import Lock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory

__all__ = []  # Publicly invisible since it's integrated into the commander

class JointRecorder:
    def __init__(self, joints=[], max_points=1000000):
        """
        Constructs a joint value recorder
        :param joints: The joints to be recorded, if the list is empty, all available joints will be returned
        :param max_points: Maximum number of trajectory points to be recorded
        """
        self.max_n_points = max_points
        self._joints = joints
        self._rec_last_point = rospy.Time(0)  # Last recorded point
        self._recording = False
        self._recorder_lock = Lock()  # Protects self.recorded_js
        self._recorded_js = []
        self._recorded_times = []
        self._recorder_rate = -1
        rospy.Subscriber("/robot/joint_states", JointState, self._cb_joint_state)

    def _cb_joint_state(self, msg):
        if len(self._recorded_js) < self.max_n_points:
            now = rospy.Time.now()
            if self._recording and self._rec_last_point + rospy.Duration(1./self._recorder_rate) < now:
                self._rec_last_point = now
                with self._recorder_lock:
                    self._recorded_js.append(msg)
                    self._recorded_times.append(msg.header.stamp)

    def recorder_start(self, rate_hz=50.):
        if self._recording:
            rospy.logerr("Already recording {}".format(str(self._joints)))
            return
        self._recorder_rate = rate_hz
        with self._recorder_lock:
            self._recorded_js = []
            self._recorded_times = []
        self._recording = True

    def recorder_stop(self, include_position=True, include_velocity=False, include_effort=False):
        """
        Stop the joint recording and returns the recorded trajectory of joints declared in the constructor
        :param include_position: if True, the joint positions will be included in the returned trajectory
        :param include_velocity:  if True, the joint velocities will be included in the returned trajectory
        :param include_effort:  if True, the joint efforts will be included in the returned trajectory
        :return: the recorded JointTrajectory
        """
        self._recording = False

        # Now we reconstruct a RobotTrajectory
        rt = RobotTrajectory()
        rt.joint_trajectory.joint_names = self._joints
        for idx, js in enumerate(self._recorded_js):
            jtp = JointTrajectoryPoint()
            if include_position:
                jtp.positions = [js.position[js.name.index(joint)] for joint in self._joints]
            if include_velocity:
                jtp.velocities = [js.velocity[js.name.index(joint)] for joint in self._joints]
            if include_effort:
                jtp.accelerations = [js.effort[js.name.index(joint)] for joint in self._joints]
            jtp.time_from_start = self._recorded_times[idx] - self._recorded_times[0]
            rt.joint_trajectory.points.append(jtp)
        return rt