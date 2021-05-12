#!/usr/bin/env python3

import math
import moveit_commander
import rospy

from geometry_msgs.msg import Pose, Twist, Vector3
from sensor_msgs.msg import LaserScan

class Movement(object):
    def __init__(self):
        rospy.init_node('movement')

        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        # self.scan_pub = rospy.Subscriber("/scan", LaserScan, self.get_scan)
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator arm
        self.move_group_arm = moveit_commander.MoveGroupCommander("arm")
        
        # the interface to the group of joints making up the turtlebot3
        # openmanipulator gripper
        self.move_group_gripper = moveit_commander.MoveGroupCommander("gripper")

        
    def move_to_ready(self):
        # move the arm
        arm_joint_goal = [0.0, 0.526, -0.142, -0.070]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_arm.stop()
        
        # move the gripper
        gripper_joint_goal = [0.01, 0.01]
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_gripper.stop()

        
    def move_to_grabbed(self):
        # move the gripper
        gripper_joint_goal = [-0.01, -0.01]
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_gripper.stop()
        
        # move the arm
        arm_joint_goal = [0, -1.08, 0.075, 0.035]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_arm.stop()
        

    def move_to_release(self):
        # move the arm
        arm_joint_goal = [0, -0.35, -0.15, 0.5]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        self.move_group_arm.stop()
        
        # move the gripper
        gripper_joint_goal = [0.01, 0.01]
        self.move_group_gripper.go(gripper_joint_goal, wait=True)
        self.move_group_gripper.stop()

        
    def run(self):
        mvmt.move_to_ready()
        rospy.sleep(2)
        mvmt.move_to_grabbed()
        rospy.sleep(2)
        mvmt.move_to_release()
        rospy.spin()

# Some Code for debugging
if __name__=="__main__":
    mvmt = Movement()
    mvmt.run()

