#!/usr/bin/env python3

import rospy, keras_ocr, cv2, cv_bridge, numpy
import numpy as np
import time

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Vector3, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image

def get_yaw_from_pose(p):
    """ A helper function that takes in a Pose object (geometry_msgs) and returns yaw"""

    yaw = (euler_from_quaternion([
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w])
            [2])

    return yaw

# Turns robot around and searches for recognizable digits.
class digit_recognition(object):

    def __init__(self):
        self.initialized = False

        rospy.init_node('digit_recognizer')
        
        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        self.pipeline = keras_ocr.pipeline.Pipeline()

        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        
        # list of the blocks positions as polar coordinate tuples (radius, angle)
        self.block_polar_coordinates = []
        # variable that tells the image call back whether to store the image
        self.capture_image = False

        self.seen_first_pose = False
        self.seen_first_scan = False
        
        self.initialized = True



    def odom_callback(self, data):
        if self.initialized:
            self.current_pose = data.pose.pose
            self.seen_first_pose = True

    def scan_callback(self, data):
        if self.initialized:
            # only search for block positions once at start
            if not self.seen_first_scan:
                # variable that tells us whether the previous angle detected something
                contiguous_object = False
                # check 180 degrees behind robot for blocks
                for angle in range(90, 271):
                    if data.ranges[angle] <= data.range_max:
                        # object detected
                        if not contiguous_object:
                            first_angle = angle
                            contiguous_object = True
                    else:
                        if contiguous_object:
                            # we are at the first angle at which an object is no longer detected
                            middle_angle = (first_angle + angle) // 2
                            self.block_polar_coordinates.append((data.ranges[middle_angle], middle_angle))
                            contiguous_object = False
                self.seen_first_scan = True

    def image_callback(self, data):
        if self.initialized:
            if self.capture_image:
                self.image_capture = data
                self.image_height = data.height
                self.image_width = data.width
                self.capture_image = False
    
    # turn the robot CCW the given angle in degrees
    def turn_clockwise(self, angle):
        turn_msg = Twist()
        turn_msg.angular = Vector3(0, 0, 0.39) # turn pi/8 rad/sec

        #wait for first pose
        while (not self.seen_first_pose):
            time.sleep(1)
        current_yaw = get_yaw_from_pose(self.current_pose)

        # convert angle to radian value in [-pi, pi]
        if angle % 360 < 180:
            angle = (angle % 360) * np.pi/180
        else:
            angle = ((angle % 360) - 360) * np.pi/180 
        destination_yaw = current_yaw + angle 

        r = rospy.Rate(30)
        # turn until we face within 1 degree of destination angle
        while abs(get_yaw_from_pose(self.current_pose) - destination_yaw) > 0.0175:
            self.movement_pub.publish(turn_msg)
            r.sleep()
        
        # Halt
        self.movement_pub.publish(Twist())
    

    # temporarily subscribes to the camera topic to get and
    # store the next available image
    def save_next_img(self):
        self.capture_image = True
        while self.capture_image:
            time.sleep(1)
    

    def search_view_for_digits(self):
        self.save_next_img()
        # take the ROS message with the image and turn it into a format cv2 can use
        img = self.bridge.imgmsg_to_cv2(self.image_capture, desired_encoding='bgr8')
        prediction_groups = self.pipeline.recognize([img])
        print("Num Predictions:", len(prediction_groups[0]))
        for (word, box) in prediction_groups[0]:
            print("Detected", word, "at box", box)


    # Executes the process of searching for the blocks, turning the robot to 
    # face them and performing digit detection
    def run_digit_search(self):
        # wait for first scan to come in
        while not self.seen_first_scan:
            time.sleep(1)
        if len(self.block_polar_coordinates) != 3:
            print("Detected:", len(self.block_polar_coordinates), " != 3 blocks, exiting")
            exit()
        # turn to face the middle_block
        self.turn_clockwise(self.block_polar_coordinates[1][1])
        # search for digits
        self.search_view_for_digits()
    

    def get_block_position(self):
        pass
        

# Some Code for debugging
if __name__=="__main__":

    print("Initializing Digit Recognition")
    dr = digit_recognition()
    print("Done Initializing, Running digit search")
    dr.run_digit_search()
    print("Finished Digit Search")