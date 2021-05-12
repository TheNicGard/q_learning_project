#!/usr/bin/env python3

import rospy, keras_ocr, cv2, cv_bridge, numpy
import numpy as np
import time, math

from collections import OrderedDict
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
class object_recognition(object):

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

        """
        The upper and lower bounds for the BGR values for the color detection (i.e.
        what is considered blue, green, and red by the robot).
        """
        self.blue_bounds = (np.array([127, 25, 25], dtype = "uint8"), np.array([255, 50, 50], dtype = "uint8"))
        self.green_bounds = (np.array([0, 100, 0], dtype = "uint8"), np.array([50, 255, 50], dtype = "uint8"))
        self.red_bounds = (np.array([0, 0, 100], dtype = "uint8"), np.array([50, 50, 255], dtype = "uint8"))

        """
        what percentage of the horizontal FOV the robot uses to determine the color
        in front of it
        """
        self.horizontal_field_percent = 0.25
        
        self.initialized = True

    def odom_callback(self, data):
        if self.initialized:
            self.current_pose = data
            self.seen_first_pose = True

    def scan_callback(self, data):
        # only search for block positions once at start
        if self.initialized and not self.seen_first_scan:
            # variable that tells us whether the previous angle detected something
            contiguous_object = False

            for angle in range(360):
                if data.ranges[angle] <= data.range_max:
                    # object detected
                    if not contiguous_object:
                        first_angle = angle
                        contiguous_object = True
                elif contiguous_object:
                    # we are at the first angle at which an object is no longer detected
                    middle_angle = ((first_angle + angle) // 2)
                    middle_angle_rad = ((first_angle + angle) // 2) * np.pi/180
                    self.block_polar_coordinates.append((data.ranges[middle_angle], middle_angle_rad))
                    contiguous_object = False
                    
            self.seen_first_scan = True
            print("Block polar coordinates:")
            for i in self.block_polar_coordinates:
                print(i)

    def image_callback(self, data):
        if self.initialized:
            if self.capture_image:
                print("Capturing Image")
                self.image_capture = data
                self.image_height = data.height
                self.image_width = data.width
                self.capture_image = False
    
    # turn the robot towards the given an angle, given in radians
    def turn_towards_target(self, angle):
        #wait for first pose
        while (not self.seen_first_pose):
            time.sleep(1)
            
        current_yaw = get_yaw_from_pose(self.current_pose.pose.pose)

        # convert angle to radian value in [-pi, pi]
        angle = math.remainder(angle , 2*np.pi)
        destination_yaw = current_yaw + angle
        if destination_yaw > np.pi:
            destination_yaw = -np.pi + (destination_yaw - np.pi)

        r = rospy.Rate(60)
        # turn until we face within 1 degree of destination angle
        while abs(get_yaw_from_pose(self.current_pose.pose.pose) - destination_yaw) > (np.pi / 180):
            # turn pi/16 rad/sec
            turn_msg = Twist()
            turn_vel = np.sign(destination_yaw - get_yaw_from_pose(self.current_pose.pose.pose)) * (np.pi / 16)
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

        if len(prediction_groups[0]) == 0:
            print("No predictions were made!")
            return
        else:
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

    def search_view_for_dumbbells(self, block_coord_index: int) -> int:
        self.save_next_img()

        # take the ROS message with the image and turn it into a format cv2 can use
        img = self.bridge.imgmsg_to_cv2(self.image_capture, desired_encoding='bgr8')
            
        blue_mask = cv2.inRange(img, self.blue_bounds[0], self.blue_bounds[1])
        green_mask = cv2.inRange(img, self.green_bounds[0], self.green_bounds[1])
        red_mask = cv2.inRange(img, self.red_bounds[0], self.red_bounds[1])
        blue_target = cv2.bitwise_and(img, img, mask=blue_mask)
        green_target = cv2.bitwise_and(img, img, mask=green_mask)
        red_target = cv2.bitwise_and(img, img, mask=red_mask)
        all_targets = cv2.bitwise_or(red_target, cv2.bitwise_or(blue_target, green_target))
            
        # check the center of the image for the most common color
        red_total, green_total, blue_total = 0, 0, 0
        img_center_w = int(img.shape[1] / 2)
        img_start_x = int(img_center_w - ((img.shape[1] * self.horizontal_field_percent) / 2))
        img_end_x = int(img_center_w + ((img.shape[1] * self.horizontal_field_percent) / 2))

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        org = (int(img.shape[0] / 2), int(img.shape[1] / 2))
        color = (255, 0, 255)
        thickness = 2
        
        for i in range(img.shape[0]):
            for j in range(img_start_x, img_end_x):
                if not np.array_equal(blue_target[i, j], [0, 0, 0]):
                    blue_total += 1
                if not np.array_equal(green_target[i, j], [0, 0, 0]):
                    green_total += 1
                if not np.array_equal(red_target[i, j], [0, 0, 0]):
                    red_total += 1

        # add prediction to the label dictionary

        if max(blue_total, green_total, red_total) == 0:
            return
        elif max(blue_total, green_total, red_total) == blue_total:
            print("detected blue dumbbell!")
            self.block_label_dictionary['b'] = self.block_polar_coordinates[block_coord_index]
        elif max(blue_total, green_total, red_total) == green_total:
            print("detected green dumbbell!")
            self.block_label_dictionary['g'] = self.block_polar_coordinates[block_coord_index]
        elif max(blue_total, green_total, red_total) == red_total:
            print("detected red dumbbell!")
            self.block_label_dictionary['r'] = self.block_polar_coordinates[block_coord_index]
            
    # Executes the process of searching for the blocks, turning the robot to 
    # face them and performing digit detection
    def run_digit_search(self):
        # wait for first scan to come in
        while not self.seen_first_scan:
            time.sleep(1)
        if len(self.block_polar_coordinates) < 3:
            raise Exception("Detected: " + str(len(self.block_polar_coordinates)) + " < 3 blocks, exiting")

        for i in range(0, len(self.block_polar_coordinates)):
            # turn towards the target
            self.turn_towards_target(self.block_polar_coordinates[i][1] - get_yaw_from_pose(self.current_pose.pose.pose))
            # search for digits within the screenshot
            self.search_view_for_digits(i)
            # search for dumbbells within the screenshot
            self.search_view_for_dumbbells(i)
            
    # should be called after run_digit_search
    # returns polar coordinates relative to the starting pose of the robot of the blocks
    def get_block_position(self, block_num:int):
        return self.block_label_dictionary[block_num]
        

# Some Code for debugging
if __name__=="__main__":

    print("Initializing Object Recognition")
    obj_rec = object_recognition()
    print("Done Initializing, Running object search")
    obj_rec.run_digit_search()
    print("Finished Object Search; Printing Label Dict")
    obj_rec.block_label_dictionary = OrderedDict(sorted(obj_rec.block_label_dictionary.items()))
    print(obj_rec.block_label_dictionary)
