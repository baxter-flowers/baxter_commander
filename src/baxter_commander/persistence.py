from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from rospy import Duration

def trajtodict( traj):
    if isinstance(traj, RobotTrajectory):
        dict_traj = {"joint_names": traj.joint_trajectory.joint_names, "points": []}
        for p in traj.joint_trajectory.points:
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
    if isinstance(state, JointState):
        return {"name": state.name, "position": state.position}
    else:
        raise TypeError("[statetodict] Waiting for a JointState input only")

def dicttostate(dic):
    js = JointState()
    js.name = dic["name"]
    js.position = dic["position"]
    return js