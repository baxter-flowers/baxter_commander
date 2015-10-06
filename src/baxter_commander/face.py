import rospy
import transformations
from numpy import arccos, isnan, pi
from baxter_interface import Head
from tf import TransformListener

__all__ = ['FaceCommander']

class FaceCommander(Head):
    def __init__(self):
        Head.__init__(self)
        self._world = 'base'
        self._tf_listener = TransformListener()
        self.set_pan(0)

    def look_at(self, obj=None):
        if obj is None:
            self.set_pan(0)
            return True

        if isinstance(obj, str):
            pose = self._tf_listener.lookupTransform('head', obj, rospy.Time(0))
        else:
            rospy.logerr("FaceCommander.look_at() accepts only strings atm")
            return False

        hyp = transformations.norm(pose)
        adj = pose[0][1]
        angle = pi/2 - arccos(float(adj)/hyp)

        if isnan(angle):
            rospy.logerr('FaceCommander cannot look at {}'.format(obj))
            return False

        self.set_pan(angle)
        return True





