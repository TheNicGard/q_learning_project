#!/usr/bin/env python3

import object_recognition

import math
import moveit_commander
import numpy as np
import rospy

from geometry_msgs.msg import Point, Pose, Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
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

        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.get_scan)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.action_sub = rospy.Subscriber('/q_learning/robot_action', Odometry, self.action_callback)
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        self.ready = False
        self.grabbed = False

        # the farthest and closest distance a robot can be to pick up an object
        self.proximity_high = 0.23
        self.proximity_low = 0.19

        # a dict to store the (x, y) positions of blocks and objects
        self.object_positions = {}

        self.initialized = True

    def odom_callback(self, data):
        if self.initialized:
            self.current_pose = data
        
    def get_scan(self, data):
        # get the 5 degrees directly in front of the robot
        front_scans = [data.ranges[0], data.ranges[1], data.ranges[2], data.ranges[358], data.ranges[359]]
        front_avg = sum(front_scans) / len(front_scans)
        a = "{:.2f}, {:.2f}, {:.2f}".format(self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y, get_yaw_from_pose(self.current_pose.pose.pose))
        # print(a)

        if self.ready:
            if self.grabbed:
                self.movement_pub.publish(Twist())
            elif front_avg > self.proximity_high:
                drive_msg = Twist()
                drive_msg.linear = Vector3(0.1, 0, 0)
                self.movement_pub.publish(drive_msg)
            elif front_avg < self.proximity_low:
                drive_msg = Twist()
                drive_msg.linear = Vector3(-0.1, 0, 0)
                self.movement_pub.publish(drive_msg)
            elif self.grabbed == False:
                self.grabbed = True
                self.movement_pub.publish(Twist())
                self.move_to_grabbed()

    def action_callback(self, data):
        dumbbell = data.robot_db
        block = data.block_id

        # current_yaw = get_yaw_from_pose(self.current_pose.pose.pose)
        # current_x = get_yaw_from_pose(self.current_pose.pose.pose)
                
                
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
        self.orient_towards_target("r")
        self.ready = True

        
    def move_to_grabbed(self):
        self.move_gripper([0.008, -0.008])        
        self.move_arm([0, -1.18, 0.225, 0.035])

        print("Dumbbell is grabbed!")
        self.grabbed = True
        self.turn_towards_target(np.pi)

        rospy.sleep(1)
        while True:
            drive_msg = Twist()
            drive_msg.linear = Vector3(0.2, 0, 0)
            self.movement_pub.publish(drive_msg)
        

    def move_to_release(self):
        self.move_arm([0, -0.35, -0.15, 0.5])
        self.move_gripper([0.01, 0.01])

        rospy.sleep(2)
        self.grabbed = False
        
        
    # turn the robot towards the given an angle, given in radians
    def turn_towards_target(self, angle):            
        current_yaw = get_yaw_from_pose(self.current_pose.pose.pose)

        # convert angle to radian value in [-pi, pi]
        angle = math.remainder(angle , 2*np.pi)
        destination_yaw = current_yaw + angle
        if destination_yaw > np.pi:
            destination_yaw = -np.pi + (destination_yaw - np.pi)

        r = rospy.Rate(60)
        # turn until we face within 1 degree of destination angle
        while abs(get_yaw_from_pose(self.current_pose.pose.pose) - destination_yaw) > (np.pi / 180):
            # turn pi/12 rad/sec
            turn_msg = Twist()
            turn_vel = np.sign(destination_yaw - get_yaw_from_pose(self.current_pose.pose.pose)) * (np.pi / 12)
            turn_msg.angular = Vector3(0, 0, turn_vel)
            self.movement_pub.publish(turn_msg)
            r.sleep()
            
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
        mvmt.move_to_ready()
        
        rospy.spin()

# Some Code for debugging
if __name__=="__main__":
    mvmt = Movement()
    
    print("Initializing Object Recognition")
    obj_rec = object_recognition.ObjectRecognition()
    print("Done Initializing, Running object search")
    obj_rec.run_digit_search()
    print("Finished Object Search; Printing Label Dict")
    print(obj_rec.block_label_dictionary)

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
