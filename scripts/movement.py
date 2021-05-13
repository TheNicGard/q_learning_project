#!/usr/bin/env python3

import math
import moveit_commander
import rospy

from geometry_msgs.msg import Pose, Twist, Vector3
from sensor_msgs.msg import LaserScan

class Movement(object):
    def __init__(self):
        rospy.init_node('movement')

        self.movement_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.get_scan)
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        self.ready = False
        self.grabbed = False

        self.proximity_high = 0.23
        self.proximity_low = 0.19

    def get_scan(self, data):
        # get the 5 degrees directly in front of the robot
        front_scans = [data.ranges[0], data.ranges[1], data.ranges[2], data.ranges[358], data.ranges[359]]
        front_avg = sum(front_scans) / len(front_scans)

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
                
        
    def move_to_ready(self):
        # move the arm
        arm_joint_goal = [0.0, 0.55, 0.3, -0.85]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_arm.stop()
        
        # move the gripper
        gripper_joint_goal = [0.01, 0.01]
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_gripper.stop()

        print("Arm ready to grab!")
        rospy.sleep(2)
        self.ready = True

        
    def move_to_grabbed(self):
        # move the gripper
        gripper_joint_goal = [-0.004, -0.004]
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_gripper.stop()
        
        # move the arm
        arm_joint_goal = [0, -1.08, 0.075, 0.035]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_arm.stop()

        print("Dumbbell is grabbed!")
        self.grabbed = True
        

    def move_to_release(self):
        # move the arm
        arm_joint_goal = [0, -0.35, -0.15, 0.5]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_arm.stop()
        
        # move the gripper
        gripper_joint_goal = [0.01, 0.01]
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_gripper.stop()

        rospy.sleep(2)
        self.grabbed = False

        
    def run(self):
        mvmt.move_to_ready()
        rospy.spin()

# Some Code for debugging
if __name__=="__main__":
    mvmt = Movement()
    mvmt.run()

