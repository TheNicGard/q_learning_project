#!/usr/bin/env python3

import rospy, keras_ocr, cv2, cv_bridge, numpy
import numpy as np
import time, math

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
        
        # list of the blocks positions as polar coordinate tuples (radius, angle) where the angle is in radians
        self.block_polar_coordinates = []
        # given a label, gives the polar coordinate tuple of the block with that label
        self.block_label_dictionary = {}
        # variable that tells the image call back whether to store the image
        self.capture_image = False

        self.seen_first_pose = False
        self.seen_first_scan = False
        
        self.initialized = True



    def odom_callback(self, data):
        if self.initialized:
            self.current_pose = data
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
                            middle_angle = ((first_angle + angle) // 2)
                            middle_angle_rad = ((first_angle + angle) // 2) * np.pi/180
                            self.block_polar_coordinates.append((data.ranges[middle_angle], middle_angle_rad))
                            contiguous_object = False
                self.seen_first_scan = True

    def image_callback(self, data):
        if self.initialized:
            if self.capture_image:
                print("Capturing Image")
                self.image_capture = data
                self.image_height = data.height
                self.image_width = data.width
                self.capture_image = False
    
    # turn the robot CCW the given an angle in radians
    def turn_clockwise(self, angle):

        #wait for first pose
        while (not self.seen_first_pose):
            time.sleep(1)
        current_yaw = get_yaw_from_pose(self.current_pose.pose.pose)

        # convert angle to radian value in [-pi, pi]
        angle = math.remainder(angle , 2*np.pi)
        destination_yaw = current_yaw + angle 

        r = rospy.Rate(60)
        # turn until we face within 1 degree of destination angle
        while abs(get_yaw_from_pose(self.current_pose.pose.pose) - destination_yaw) > 0.0175:
            turn_msg = Twist()
            abs(get_yaw_from_pose(self.current_pose.pose.pose) - destination_yaw) 
            # turn pi/16 rad/sec
            turn_vel = np.sign(destination_yaw - get_yaw_from_pose(self.current_pose.pose.pose)) * 0.196
            turn_msg.angular = Vector3(0, 0, turn_vel)
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
    
    
    # saves an image and searches it for digits, adding the recognized digits to
    # the label dictionary with value according index passed through
    def search_view_for_digits(self, block_coord_index: int) -> int:
        self.save_next_img()

        # take the ROS message with the image and turn it into a format cv2 can use
        img = self.bridge.imgmsg_to_cv2(self.image_capture, desired_encoding='bgr8')
        
        prediction_groups = self.pipeline.recognize([img])
        print("Num Predictions:", len(prediction_groups[0]))

        # calculate distance to center of image for each prediction
        center_list = []
        for (word, box) in prediction_groups[0]:
            cx = (box[0][0] + box[1][0] + box[2][0] + box[3][0])/4
            cy = (box[0][1] + box[1][1] + box[2][1] + box[3][1])/4
            center_distance_squared = (cx - self.image_width/2) ** 2 + (cy - self.image_height/2) ** 2
            center_list.append((word, center_distance_squared))

        # initialize minimum observation as first in list
        closest_center_index = 0
        # get index of best centered prediction
        for i in range(len(center_list)):
            if center_list[i][1] < center_list[closest_center_index][1]:
                closest_center_index = i
        print(center_list)
        print(center_list[closest_center_index][1])
        # add prediction to the label dictionary
        self.block_label_dictionary[center_list[closest_center_index][0]] = self.block_polar_coordinates[block_coord_index]


    # Executes the process of searching for the blocks, turning the robot to 
    # face them and performing digit detection
    def run_digit_search(self):
        # wait for first scan to come in
        while not self.seen_first_scan:
            time.sleep(1)
        if len(self.block_polar_coordinates) != 3:
            print("Detected:", len(self.block_polar_coordinates), " != 3 blocks, exiting")
            exit()
        # turn to face the first block
        self.turn_clockwise(self.block_polar_coordinates[0][1])
        # search for digits
        self.search_view_for_digits(0)
        # turn to face the second block
        self.turn_clockwise(self.block_polar_coordinates[1][1] - get_yaw_from_pose(self.current_pose.pose.pose))
        # search for digits
        self.search_view_for_digits(1)
        # update dictionary (last block must have missing label)
        if 1 not in self.block_label_dictionary:
            self.block_label_dictionary['1'] = self.block_polar_coordinates[2]
        elif 2 not in self.block_label_dictionary:
            self.block_label_dictionary['2'] = self.block_polar_coordinates[2]
        else:
            self.block_label_dictionary['3'] = self.block_polar_coordinates[2]


    # should be called after run_digit_search
    # returns polar coordinates relative to the starting pose of the robot of the blocks
    def get_block_position(self, block_num:int):
        return self.block_label_dictionary[block_num]
        

# Some Code for debugging
if __name__=="__main__":

    print("Initializing Digit Recognition")
    dr = digit_recognition()
    print("Done Initializing, Running digit search")
    dr.run_digit_search()
    print("Finished Digit Search; Printing Label Dict")
    print(dr.block_label_dictionary)