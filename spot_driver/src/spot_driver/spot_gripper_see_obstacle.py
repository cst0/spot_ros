#!/usr/bin/env python3

from math import pi, floor
from numpy import average
import cv_bridge
import rospy
from sensor_msgs.msg import Range, Image

class SpotGripperObstacleDetector:
    def __init__(self):
        self.gripper_depth_image_sub = rospy.Subscriber('/spot/camera/hand_depth/image', Image, self.gripper_depth_image_callback, queue_size=1)
        self.gripper_laserscan_pub = rospy.Publisher('/spot/camera/hand_depth/range', Range, queue_size=1)
        self.bridge = cv_bridge.CvBridge()

    def gripper_depth_image_callback(self, image_msg:Image):
        self.gripper_depth_image_msg = image_msg
        self.gripper_depth_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        if self.gripper_depth_image is not None:
            self.do_gripper_laserscan()

    def do_gripper_laserscan(self):
        range_msg = Range()
        range_msg.header.frame_id = self.gripper_depth_image_msg.header.frame_id
        range_msg.header.stamp = self.gripper_depth_image_msg.header.stamp
        range_msg.radiation_type = Range.INFRARED
        range_msg.field_of_view = pi / 2.0

        range_msg.min_range = 0.0
        range_msg.max_range = 1.25

        shape = self.gripper_depth_image.shape
        center_x = floor(shape[1] / 2)
        center_y = floor(shape[0] / 2)
        query_pixels = [(center_x - i, center_y - j) for i in range(-2, 2) for j in range(-2, 2)]
        values = []
        for pixel in query_pixels:
            d = self.get_depth_at_pixel(pixel)
            if d is not None:
                values.append(d)
        range_msg.range = average(values) / 1000.0 # convert to meters from mm
        self.gripper_laserscan_pub.publish(range_msg)

    def get_depth_at_pixel(self, pixel):
        if (
                pixel[1] < 0 or
                pixel[1] >= self.gripper_depth_image.shape[1] or
                pixel[0] < 0 or
                pixel[0] >= self.gripper_depth_image.shape[0]):
            return None
        return self.gripper_depth_image[pixel[1], pixel[0]]
