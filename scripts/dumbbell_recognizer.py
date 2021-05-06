#!/usr/bin/env python3

from geometry_msgs.msg import Twist, Vector3
import numpy as np
import rospy, rospkg, cv2, cv_bridge, numpy
from sensor_msgs.msg import Image, LaserScan

class DumbbellRecognizer(object):

    def __init__(self):

        self.initalized = False

        rospy.init_node('cat_recognizer')

        # subscribe to the robot's RGB camera data stream
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)

        self.image_rot = 0
        
        self.lidar_sub = rospy.Subscriber('scan', LaserScan, self.robot_scan_received)
        
        self.move_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # set up ROS / cv bridge
        self.bridge = cv_bridge.CvBridge()

        # load the opencv2 XML classifier for cat faces
        # obtained from https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface_extended.xml
        rp = rospkg.RosPack()
        self.catface_cascade = cv2.CascadeClassifier(
            rp.get_path('class_meeting_11_cat_recognition') + '/scripts/catface_detector.xml')

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

        self.initalized = True


    # color detection based on implementation here:
    # https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
    def image_callback(self, data):

        if (not self.initalized):
            return

        if (not self.seen_first_image):

            # we have now seen the first image
            self.seen_first_image = True

            # take the ROS message with the image and turn it into a format cv2 can use
            img = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            
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
            
            for i in range(img.shape[0]):
                for j in range(img_start_x, img_end_x):
                    if not np.array_equal(blue_target[i, j], [0, 0, 0]):
                        blue_total += 1
                    if not np.array_equal(green_target[i, j], [0, 0, 0]):
                        green_total += 1
                    if not np.array_equal(red_target[i, j], [0, 0, 0]):
                        red_total += 1

            if max(blue_total, green_total, red_total) == 0:
                print("The robot is facing no dumbell.")
            elif max(blue_total, green_total, red_total) == blue_total:
                print("The robot is facing the blue dumbell.")
            elif max(blue_total, green_total, red_total) == green_total:
                print("The robot is facing the green dumbell.")
            elif max(blue_total, green_total, red_total) == red_total:
                print("The robot is facing the red dumbell.")
            else:
                print("The robot is facing the blue dumbell (BGR: {0}, {1}, {2}).".format(str(blue_total), str(green_total), str(red_total)))

            all_targets = cv2.rectangle(all_targets, (img_start_x, 0), (img_end_x, img.shape[0]), (0, 255, 255), 1)
            # visualize the robot's target in the image
            cv2.imshow('img', all_targets)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def robot_scan_received(self, data):
        return

    def run(self):
        
        rospy.spin()


if __name__ == '__main__':
    node = DumbbellRecognizer()
    node.run()
