import rospy
from std_msgs.msg import Float32
from threading import Thread, RLock
from numpy import sin

class Halo(object):
    def __init__(self):
        self._green_pub = rospy.Publisher("/robot/sonar/head_sonar/lights/set_green_level", Float32, queue_size=1)
        self._red_pub = rospy.Publisher("/robot/sonar/head_sonar/lights/set_red_level", Float32, queue_size=1)
        self._red = 0
        self._green = 100
        self._flashing_lock = RLock()
        self._flashing_speed = 20
        self._flashing = False
        self._flashing_thread = None
        self._flashing_color = 'green'
    
    @property
    def red(self):
        return self._red

    @property
    def green(self):
        return self._green

    @red.setter
    def red(self, value):
        self._red = max(min(value, 100), 0)
        self._red_pub.publish(Float32(data=self._red))

    @green.setter
    def green(self, value):
        self._green = max(min(value, 100), 0)
        self._green_pub.publish(Float32(data=self._green))

    def set_yellow(self):
        with self._flashing_lock:
            if not self._flashing:
                self.green = 90
                self.red = 60

    def set_green(self):
        with self._flashing_lock:
            if not self._flashing:
                self.green = 100
                self.red = 0

    def set_red(self):
        with self._flashing_lock:
            if not self._flashing:
                self.green = 0
                self.red = 100

    def set_off(self):
        with self._flashing_lock:
            if not self._flashing:
                self.green = 0
                self.red = 0
        
    def start_flashing(self, color='yellow', speed=20):
        with self._flashing_lock:
            if not self._flashing:
                self._flashing_speed = speed
                self._flashing_color = color
                self._flashing = True
                self._flashing_thread = Thread(target=self._threaded_flashing)
                self._flashing_thread.setDaemon(True)
                self._flashing_thread.start()

    def _threaded_flashing(self):
        self.set_off()
        def generate_flash():
            num_steps = 31
            while True:
                for color in range(0, num_steps, 2):
                    yield int(sin(color/10.)*100)
                for color in range(num_steps, 0, -2):
                    yield int(sin(color/10.)*100)

        flash = generate_flash()
        rate = rospy.Rate(self._flashing_speed)
        while self._flashing and not rospy.is_shutdown():
            color = flash.next()
            if self._flashing_color in ['red', 'yellow']:
                self.red = color
            if self._flashing_color in ['green', 'yellow']:
                self.green = color
            rate.sleep()

    def stop_flashing(self):
        with self._flashing_lock:
            if self._flashing:
                self._flashing = False
        self.set_off()

if __name__ == '__main__':
    rospy.init_node('test_halo')
    halo = Halo()
    rospy.sleep(1)
    halo.start_flashing('red')
    rospy.sleep(1)
    halo.set_green()  # Unauthorized
    rospy.sleep(2)
    halo.stop_flashing()
    rospy.sleep(5)
    halo.set_yellow()