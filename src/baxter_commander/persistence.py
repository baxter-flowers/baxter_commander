from moveit_msgs.msg import RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from transformations import pose_to_list, list_to_pose
from rospy import Duration, Time

__all__ = ['trajtodict', 'dicttotraj', 'statetodict', 'dicttostate', 'pathtodict', 'dicttopath']

def trajtodict(traj):
    if isinstance(traj, RobotTrajectory):
        traj = traj.joint_trajectory
    if isinstance(traj, JointTrajectory):
        dict_traj = {"joint_names": traj.joint_names, "points": []}
        for p in traj.points:
            d = {"positions":p.positions, "time_from_start":p.time_from_start.to_sec()}
            dict_traj["points"].append(d)
        return dict_traj
    else:
        raise TypeError("[trajtodict] Waiting for a RobotTrajectory input only")

def dicttotraj(dic):
    rt = RobotTrajectory()
    rt.joint_trajectory.joint_names = dic["joint_names"]
    for p in dic["points"]:
        jtp = JointTrajectoryPoint()
        jtp.positions = p["positions"]
        jtp.time_from_start = Duration(p["time_from_start"])
        rt.joint_trajectory.points.append(jtp)
    return rt

def statetodict(state):
    if isinstance(state, RobotState):
        state = state.joint_state
    if isinstance(state, JointState):
        return {"name": state.name, "position": state.position}
    else:
        raise TypeError("[statetodict] Waiting for a JointState input only")

def dicttostate(dic):
    rs = RobotState()
    rs.joint_state.name = dic["name"]
    rs.joint_state.position = dic["position"]
    return rs

def pathtodict(path):
    assert isinstance(path, Path)
    frame_id = path.header.frame_id
    if frame_id == '' and len(path.poses) > 0:
        frame_id = path.poses[0].header.frame_id

    dic = {"frame_id": frame_id, "points": []}
    for point in path.poses:
        dic["points"].append({"time": point.header.stamp.to_sec(), "pose": pose_to_list(point)})
    return dic

def dicttopath(dic):
    path = Path()
    path.header.frame_id = dic["frame_id"]
    for point in dic["points"]:
        path.poses.append(list_to_pose(point["pose"], frame_id=dic["frame_id"]))
        path.poses[-1].header.stamp = Time(point["time"])
    return path
