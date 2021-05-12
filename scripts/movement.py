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
        """
        attempt using task space
        """
        pose_goal = Pose()
        pose_goal.orientation.w = 0.0
        pose_goal.position.x = 0.28
        pose_goal.position.y = 0.0
        pose_goal.position.z = 0.19
        # self.move_group_arm.set_pose_target(pose_goal)
        # self.move_group_arm.go(wait=True)
        # self.move_group_arm.stop()


        """
        attempt using joint space
        """
        arm_joint_goal = [0.0, 0.555, 0.9, -0.392]
        self.move_group_arm.go(arm_joint_goal, wait=True)
        
        # move the gripper
        gripper_joint_goal = [0.05, 0.05]
        # self.move_group_gripper.go(gripper_joint_goal, wait=True)
        # self.move_group_gripper.stop()

# Some Code for debugging
if __name__=="__main__":
    mvmt = Movement()
    mvmt.move_to_ready()

