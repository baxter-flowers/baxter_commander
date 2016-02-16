import rospy
from std_msgs.msg import Float32

class Halo(object):
    def __init__(self):
        self.green = rospy.Publisher("/robot/sonar/head_sonar/lights/set_green_level", Float32, queue_size = 1)
        self.red = rospy.Publisher("/robot/sonar/head_sonar/lights/set_red_level", Float32, queue_size = 1)
        
    def set_yellow(self):
        self.green.publish(Float32(data=90))
        self.red.publish(Float32(data=60))

    def set_green(self):
        self.green.publish(Float32(data=100))
        self.red.publish(Float32(data=0))

    def set_red(self):
        self.green.publish(Float32(data=0))
        self.red.publish(Float32(data=100))

    def set_off(self):
        self.green.publish(Float32(data=0))
        self.red.publish(Float32(data=0))

