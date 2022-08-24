#!/usr/bin/env python3

from math import pi, floor
from numpy import average
import cv_bridge
import rospy
from sensor_msgs.msg import Range, Image


class SpotGripperObstacleDetector:
    def __init__(self):
        # fmt:off
        self.gripper_depth_image_sub        = rospy.Subscriber('/spot/camera/hand_depth/image', Image, self.gripper_depth_image_callback, queue_size = 1)

        self.gripper_laserscan_pub_left     = rospy.Publisher('/spot/camera/hand_depth/ranges/left', Range, queue_size = 1)
        self.gripper_laserscan_pub_midleft  = rospy.Publisher('/spot/camera/hand_depth/ranges/midleft', Range, queue_size = 1)
        self.gripper_laserscan_pub          = rospy.Publisher('/spot/camera/hand_depth/range', Range, queue_size = 1)
        self.gripper_laserscan_pub_right    = rospy.Publisher('/spot/camera/hand_depth/ranges/right', Range, queue_size = 1)
        self.gripper_laserscan_pub_midright = rospy.Publisher('/spot/camera/hand_depth/ranges/midright', Range, queue_size = 1)
        self.bridge = cv_bridge.CvBridge()
        # fmt:on

    def gripper_depth_image_callback(self, image_msg: Image):
        self.gripper_depth_image_msg = image_msg
        self.gripper_depth_image = self.bridge.imgmsg_to_cv2(
            image_msg, desired_encoding="passthrough"
        )
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

        min_x = 3
        max_x = shape[1] - 3
        center_x = floor(shape[1] / 2)
        quarter_x = floor(shape[1] / 4)
        center_y = floor(shape[0] / 2)

        x_scans = {
            self.gripper_laserscan_pub_left     : (min_x              , center_y) ,
            self.gripper_laserscan_pub_midleft  : (quarter_x          , center_y) ,
            self.gripper_laserscan_pub          : (center_x           , center_y) ,
            self.gripper_laserscan_pub_midright : (center_x+quarter_x , center_y) ,
            self.gripper_laserscan_pub_right    : (max_x              , center_y) ,
        }

        for pub, (x, y) in x_scans.items():
            range_msg.range = self.get_avg_range(x, y)
            pub.publish(range_msg)

    def get_avg_range(self, center_x, center_y):
        query_pixels = [
            (center_x - i, center_y - j) for i in range(-2, 2) for j in range(-2, 2)
        ]
        values = []
        for pixel in query_pixels:
            d = self.get_depth_at_pixel(pixel)
            if d is not None:
                values.append(d)
        _range = average(values) / 1000.0  # convert to meters from mm
        return _range

    def get_depth_at_pixel(self, pixel):
        if (
            pixel[1] < 0
            or pixel[1] >= self.gripper_depth_image.shape[1]
            or pixel[0] < 0
            or pixel[0] >= self.gripper_depth_image.shape[0]
        ):
            return None
        return self.gripper_depth_image[pixel[1], pixel[0]]
