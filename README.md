# Baxter Commander
Custom improvements to the baxter_interface package to control Baxter at a higher level.
Baxter Commander features are:

## Trajectory preview before execution
This feature allows to display motions without executing them nor simulating the whole robot and environment. This is a good compromise between simulation and real-robot execution. A flag `display_only=True` can be passed to execution methods to simulate trajectories. This python flag allows flexibility to choose what motions can be actually executed and what motions need to be checked while coding.

## Simple trajectory operations
 * Generation of motions from a start cartesian point or robot state to a goal cartesian point or robot state, with trapezoidal-shaped speed and acceleration.
 * Generation of reverse trajectories
 * Generation of cartesian motions using sequential IK calls (same as MoveIt's feature with allowance of IK failures within the path)

## Trajectory recording, dumping and loading
In memory trajectories are handled via MoveIt-compatible objects (`RobotState`, `RobotTrajectory`) or ROS-compatible objects (`JointState`, `JointTrajectory`). Both state and trajectories can be dumped into files for further loading thanks to functions in the `persistence` module.

## Enrichment of transformation toolbox `tf.transformations`
The `transformations` module extends `tf`'s one with additional functions to invert or multiply transforms, compute norms, distances, convert the list format `[[x, y, z], [x, y, z, w]]` from/to `geometry_msgs.PoseStamped` or matrices 4x4.

## Example usage
Start the background service and open an ipython notebook with the [example notebook](https://github.com/baxter-flowers/baxter_commander/blob/master/notebooks/commander.ipynb): 
```
roslaunch baxter_commander commander.launch # gui:=false to disable motion preview in RViz
cd baxter_commander/notebooks
ipython notebook
```

## Installation procedure
Download `baxter_commander` in your ROS workspace src directory `~/ros_ws/src/`
```
sudo apt-get install ros-indigo-moveit-full python-pip
sudo pip install xmltodict
cd ~ros_ws/
catkin_make
```
