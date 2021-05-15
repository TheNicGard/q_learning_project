#!/usr/bin/env python3

import object_recognition

import math
import moveit_commander
import numpy as np
import rospy, cv2, cv_bridge

from geometry_msgs.msg import Point, Pose, Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from tf.transformations import quaternion_from_euler, euler_from_quaternion

"""
A helper function that takes in a Pose object (geometry_msgs) and returns yaw
"""
def get_yaw_from_pose(p):
    yaw = (euler_from_quaternion([
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w])
            [2])

    return yaw

class Movement(object):
    def __init__(self):
        # rospy.init_node('movement')
        # seems to have a problem with also initializing object_recognition?
        self.initialized = False
        self.target_driving_towards = None
        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.get_scan)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.action_sub = rospy.Subscriber('/q_learning/robot_action', Odometry, self.action_callback)
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)

        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        self.seen_first_image = False

        """
        The upper and lower bounds for the BGR values for the color detection (i.e.
        what is considered blue, green, and red by the robot).
        """
        self.blue_bounds = (np.array([100, 0, 0], dtype = "uint8"), np.array([255, 50, 50], dtype = "uint8"))
        self.green_bounds = (np.array([0, 100, 0], dtype = "uint8"), np.array([50, 255, 50], dtype = "uint8"))
        self.red_bounds = (np.array([0, 0, 100], dtype = "uint8"), np.array([50, 50, 255], dtype = "uint8"))

        """
        what percentage of the horizontal FOV the robot uses to determine the color
        in front of it
        """
        self.horizontal_field_percent = 0.25
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        self.scan_complete = False
        self.driving = False
        self.on_way_to_block = False

        self.ranges = []

        # the farthest and closest distance a robot can be to pick up an object
        self.db_proximity_high = 0.23
        self.db_proximity_low = 0.19
        self.block_proximity_high = 0.55
        self.block_proximity_low = 0.5

        # a dict to store the (x, y) positions of blocks and objects
        self.object_positions = {}
        
        self.driving_angular_velocity = 0

        self.action_queue = []

        self.initialized = True

    def odom_callback(self, data):
        if self.initialized:
            self.current_pose = data
        
    def get_scan(self, data):
        # get the 5 degrees directly in front of the robot
        front_scans = data.ranges[0:3] + data.ranges[358:360]
        front_avg = sum(front_scans) / len(front_scans)

        # the proximity required changes depending on if the robot is travelling
        # towards a dumbbell or a block
        prox_hi, prox_lo = 0, 0
        if self.on_way_to_block:
            prox_hi, prox_lo = self.block_proximity_high, self.block_proximity_low
        else:
            prox_hi, prox_lo = self.db_proximity_high, self.db_proximity_low
            
        # drive towards target using LiDAR
        if self.driving:
            drive_msg = Twist()
            if front_avg > prox_hi:
                drive_msg.linear = Vector3(0.1, 0, 0)
                if self.target_driving_towards in self.color_turn_dict:
                    print("adjusting drive to", self.target_driving_towards, " by", 0.001*self.color_turn_dict[self.target_driving_towards])
                    drive_msg.angular = Vector3(0, 0, 0.001*self.color_turn_dict[self.target_driving_towards])
            elif front_avg < prox_lo:
                drive_msg.linear = Vector3(-0.1, 0, 0)
                if self.target_driving_towards in self.color_turn_dict:
                    print("adjusting drive to", self.target_driving_towards, " by", 0.001*self.color_turn_dict[self.target_driving_towards])
                    drive_msg.angular = Vector3(0, 0, 0.001*self.color_turn_dict[self.target_driving_towards])
            else:
                self.driving = False
            self.movement_pub.publish(drive_msg)

    
    def image_callback(self, data):

        if self.initialized:
            if (not self.seen_first_image):
                # we have now seen the first image
                self.seen_first_image = True
            # take the ROS message with the image and turn it into a format cv2 can use
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            blue_mask = cv2.inRange(img, self.blue_bounds[0], self.blue_bounds[1])
            green_mask = cv2.inRange(img, self.green_bounds[0], self.green_bounds[1])
            red_mask = cv2.inRange(img, self.red_bounds[0], self.red_bounds[1])

            Mred = cv2.moments(red_mask)
            Mgreen = cv2.moments(green_mask)
            Mblue = cv2.moments(blue_mask)
            h, w, d = img.shape
            # direction to turn for a given color
            self.color_turn_dict = {}
            # green center
            if Mgreen['m00'] > 0:
                cx = int(Mgreen['m10']/Mgreen['m00'])
                cy = int(Mgreen['m01']/Mgreen['m00'])
                # circle for debugging window
                #cv2.circle(img, (cx, cy), 20, (0,0,255), -1)
                self.color_turn_dict["green"] = (w/2 - cx)
            else:
                self.color_turn_dict["green"] = 0
            # red center
            if Mred['m00'] > 0:
                cx = int(Mred['m10']/Mred['m00'])
                cy = int(Mred['m01']/Mred['m00'])
                #cv2.circle(img, (cx, cy), 20, (0,0,255), -1)
                self.color_turn_dict["red"] = (w/2 - cx)
            else:
                self.color_turn_dict["red"] = 0
            # blue center
            if Mblue['m00'] > 0:
                cx = int(Mblue['m10']/Mblue['m00'])
                cy = int(Mblue['m01']/Mblue['m00'])
                #cv2.circle(img, (cx, cy), 20, (0,0,255), -1)
                self.color_turn_dict["blue"] = (w/2 - cx)
            else:
                self.color_turn_dict["blue"] = 0
            
            # show the debugging window
            #cv2.imshow("window", img)
            #cv2.waitKey(3)

    # moves the given action to the block    
    def execute_action(self, action):
        dumbbell = action[0]
        block = action[1]
        self.orient_towards_target(dumbbell)
        self.move_to_ready()
        self.driving = True
        self.target_driving_towards = dumbbell
        while self.driving:
            pass
    
        self.move_to_grabbed()
        self.orient_towards_target(block)
        self.on_way_to_block = True
        self.driving = True
        while self.driving:
            pass
        self.on_way_to_block = False
        
        self.move_to_release()
        
    
    def action_callback(self, data):
        self.action_queue.append((data.robot_db, data.block_id))


    def temp_callback(self):
        actions = [("green", "1"), ("red", "3"), ("blue", "2")]

        for (dumbbell, block) in actions:
            self.orient_towards_target(dumbbell)
            self.move_to_ready()
            self.driving = True
            self.target_driving_towards = dumbbell
            while self.driving:
                pass
        
            self.move_to_grabbed()
            self.orient_towards_target(block)
            self.on_way_to_block = True
            self.driving = True
            while self.driving:
                pass
            self.on_way_to_block = False
            
            self.move_to_release()
                
                
    def move_arm(self, goal):
        self.move_group_arm.go(goal, wait=True)
        self.move_group_arm.stop()

        
    def move_gripper(self, goal):
        self.move_group_gripper.go(goal, wait=True)
        self.move_group_gripper.stop()

                
    def move_to_ready(self):
        self.move_arm([0.0, 0.55, 0.3, -0.85])
        self.move_gripper([0.018, 0.018])

        print("Arm ready to grab!")
        rospy.sleep(2)

        
    def move_to_grabbed(self):
        self.move_gripper([0.008, -0.008])        
        self.move_arm([0, -1.18, 0.225, 0.035])

        print("Dumbbell is grabbed!")
        rospy.sleep(2)
        

    def move_to_release(self):
        self.move_arm([0, -0.35, -0.15, 0.5])
        self.move_gripper([0.01, 0.01])

        print("Dumbbell has been released!")
        rospy.sleep(2)
        
        
    # turn the robot towards the given an angle, given in radians
    def turn_towards_target(self, angle):            
        current_yaw = get_yaw_from_pose(self.current_pose.pose.pose)

        # convert angle to radian value in [-pi, pi]
        angle = math.remainder(angle , 2*np.pi)
        
        destination_yaw = current_yaw + angle
        if destination_yaw > np.pi:
            destination_yaw -= (2 * np.pi)
        elif destination_yaw < -np.pi:
            destination_yaw += (2 * np.pi)

        r = rospy.Rate(60)
        # turn until we face within 1 degree of destination angle
        while abs(get_yaw_from_pose(self.current_pose.pose.pose) - destination_yaw) > (np.pi / 180):
            # turn pi/12 rad/sec
            turn_msg = Twist()
            turn_vel = np.sign(destination_yaw - get_yaw_from_pose(self.current_pose.pose.pose)) * (np.pi / 12)
            turn_msg.angular = Vector3(0, 0, turn_vel)
            self.movement_pub.publish(turn_msg)
            r.sleep()
            print("{0}/{1}: {2}".format(str(get_yaw_from_pose(self.current_pose.pose.pose)), str(destination_yaw), str(abs(get_yaw_from_pose(self.current_pose.pose.pose) - destination_yaw))))
            
        # Halt
        self.movement_pub.publish(Twist())


    # given a key of an object, turn the robot towards that object
    def orient_towards_target(self, key):
        object_pos = self.object_positions[key]
        if object_pos is None:
            raise Exception("key \"{0}\" not found!".format(key))

        robot_pos = self.current_pose.pose.pose.position
        x_dist = object_pos.x - robot_pos.x
        y_dist = object_pos.y - robot_pos.y
        angle = np.arctan(y_dist / x_dist)
        """
        account for using arctan; angle changes based on sign of distances (?)
        I checked the math, and i'm 98% sure this is correct
        """
        if np.sign(x_dist) == -1:
            if np.sign(y_dist) == -1:
                angle += np.pi
            else:
                angle = np.pi + angle
        elif np.sign(y_dist) == -1:
            angle += (np.pi * 2)
            
        print("Turning towards " + key + " at {:.2f}".format(angle))
        self.turn_towards_target(angle - get_yaw_from_pose(self.current_pose.pose.pose))

        
    def run(self):
        #self.temp_callback()
        queued_action_index = 0
        rospy.spin()
        while not rospy.is_shutdown:
            try:
                self.execute_action(self.action_queue[queued_action_index])
                queued_action_index += 1
            except:
                print("ran out of actions in queue")
            queued_action_index += 1

# Some Code for debugging
if __name__=="__main__":
    mvmt = Movement()
    print("Initializing Object Recognition")
    obj_rec = object_recognition.ObjectRecognition()
    print("Done Initializing, Running object search")
    obj_rec.run_digit_search()
    print("Finished Object Search; Printing Label Dict")
    print(obj_rec.block_label_dictionary)
    mvmt.scan_complete = True

    # convert the (distance, angle) to [x, y, z] (where Z doesn't matter)
    for (key, value) in obj_rec.block_label_dictionary.items():
        distance, angle = value[0], value[1]
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        mvmt.object_positions[key] = Point(x, y, 0)

    for (key, value) in mvmt.object_positions.items():
        print("{0}:\n{1}".format(key, str(value)))

    a = "{:.2f}, {:.2f}, {:.2f}".format(mvmt.current_pose.pose.pose.position.x, mvmt.current_pose.pose.pose.position.y, get_yaw_from_pose(mvmt.current_pose.pose.pose))
    print(a)

    mvmt.run()
