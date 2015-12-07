#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup()
d['packages'] = ['baxter_commander', 'transformations']
d['package_dir'] = {'': 'src'}
d['requires'] = ['baxter_pykdl', 'baxter_interface', 'baxter_core_msgs', 'actionlib_msgs', 'moveit_msgs', 'trajectory_msgs', 'control_msgs', 'geometry_msgs']
d['install_requires'] = ['xmltodict']
setup(**d)
