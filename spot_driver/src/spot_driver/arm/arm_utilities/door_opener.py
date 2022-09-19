# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).
#
# This code was originally the Boston Dynamics 'arm_door' example, which I've modified for more general use.
# --cst (Chris Thierauf, chris@cthierauf.com)

import math
import time
from typing import List
import rospy

from bosdyn.api import manipulation_api_pb2
from bosdyn.api import basic_command_pb2
from bosdyn.api import geometry_pb2
from bosdyn.api.manipulation_api_pb2 import (
    ManipulationApiFeedbackRequest,
    ManipulationApiRequest,
    WalkToObjectInImage,
)
from bosdyn.api.spot import door_pb2
from bosdyn.client import frame_helpers
from bosdyn.client.door import DoorClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient

import cv2
import numpy as np
from vision_msgs.msg import Detection2D


def check_estop(robot):
    """Verify that robot is not estopped. E-Stop should be run in a separate process.

    Args:
        robot: (Robot) Interface to Spot robot.
    """
    assert not robot.is_estopped(), (
        "Robot is estopped. Please use an external E-Stop client, "
        "such as the estop SDK example, to configure E-Stop."
    )


def walk_to_object_in_image(robot, request_manager: "RequestManager", debug):
    """Command the robot to walk toward user selected point. The WalkToObjectInImage feedback
    reports a raycast result, converting the 2D touchpoint to a 3D location in space.

    Args:
        robot: (Robot) Interface to Spot robot.
        request_manager: (RequestManager) Object for bookkeeping user touch points.
        debug (bool): Show intermediate debug image..

    Returns:
        ManipulationApiResponse: Feedback from WalkToObjectInImage request.
    """
    manip_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    manipulation_api_request = request_manager.get_walk_to_object_in_image_request(
        debug
    )

    # Send a manipulation API request. Using the points selected by the user, the robot will
    # walk toward the door handle.
    rospy.loginfo("Walking toward door...")
    response = manip_client.manipulation_api_command(manipulation_api_request)

    # Check feedback to verify the robot walks to the handle. The service will also return a
    # FrameTreeSnapshot that contain a walkto_raycast_intersection point.
    command_id = response.manipulation_cmd_id
    feedback_request = ManipulationApiFeedbackRequest(manipulation_cmd_id=command_id)
    timeout_sec = 15.0
    end_time = time.time() + timeout_sec
    while time.time() < end_time:
        response = manip_client.manipulation_api_feedback_command(feedback_request)
        assert (
            response.manipulation_cmd_id == command_id
        ), "Got feedback for wrong command."
        if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            return response
    raise Exception("Manip command timed out. Try repositioning the robot.")


def get_images_as_cv2(robot, sources):
    """Request image sources from robot. Decode and store as OpenCV image as well as proto.

    Args:
        robot: (Robot) Interface to Spot robot.
        sources: (list) String names of image sources.

    Returns:
        dict: Dictionary from image source name to (image proto, CV2 image) pairs.
    """
    image_client = robot.ensure_client(ImageClient.default_service_name)
    image_responses = image_client.get_image_from_sources(sources)
    image_dict = dict()
    for response in image_responses:
        # Convert image proto to CV2 image, for display later.
        image = np.frombuffer(response.shot.image.data, dtype=np.uint8)
        image = cv2.imdecode(image, -1)
        image_dict[response.source.name] = (response, image)
    return image_dict


class RequestManager:
    """Helper object for displaying side by side images to the user and requesting user selected
    touchpoints. This class handles the bookkeeping for converting between a touchpoints of side by
    side display image of the frontleft and frontright fisheye images and the individual images.

    Args:
        image_dict: (dict) Dictionary from image source name to (image proto, CV2 image) pairs.
        window_name: (str) Name of display window..
    """

    def __init__(self):
        self.detections = []
        self.image_dict = {}
        self.window_name=''
        self.handle_position_side_by_side= None
        self.hinge_position_side_by_side=None
        self._side_by_side=None
        self.image_detection_source_name=None
        self._ready = False

    def init_ros_image_mode(self, detections: List[Detection2D]):
        self.detections = detections
        self.mode = "ros"
        self._ready = True

    def init_user_input_mode(self, image_dict, window_name):
        self.image_dict = image_dict
        self.window_name = window_name
        self.handle_position_side_by_side = None
        self.hinge_position_side_by_side = None
        self._side_by_side = None
        self.image_detection_source_name = None
        self.mode = "user"
        self._ready = True

    @property
    def source_image(self):

        if not self._ready:
            raise Exception(
                "Class not instantiated via ros_image_mode or user_input_mode"
            )

        if self.mode == "user":
            return self.generate_side_by_side()
        if self.mode == "ros":
            return self.generate_ros_source()

    def generate_side_by_side(self):
        if self._side_by_side:
            return self._side_by_side

        # Convert PIL images to numpy for processing.
        fr_fisheye_image = self.image_dict["frontright_fisheye_image"][1]
        fl_fisheye_image = self.image_dict["frontleft_fisheye_image"][1]

        # Rotate the images to align with robot Z axis.
        fr_fisheye_image = cv2.rotate(fr_fisheye_image, cv2.ROTATE_90_CLOCKWISE)
        fl_fisheye_image = cv2.rotate(fl_fisheye_image, cv2.ROTATE_90_CLOCKWISE)

        self._side_by_side = np.hstack([fr_fisheye_image, fl_fisheye_image])

        return self._side_by_side

    def generate_ros_source(self):
        return self.ros_image

    def user_input_set(self):
        """bool: True if handle and hinge position set."""
        return self.handle_position_side_by_side and self.hinge_position_side_by_side

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.handle_position_side_by_side:
                cv2.circle(self.source_image, (x, y), 30, (255, 0, 0), 5)
                _draw_text_on_image(self.source_image, "Click hinge.")
                cv2.imshow(self.window_name, self.source_image)
                self.handle_position_side_by_side = (x, y)
            elif not self.hinge_position_side_by_side:
                self.hinge_position_side_by_side = (x, y)

    def get_user_input_handle_and_hinge(self):
        """Open window showing the side by side fisheye images with on screen prompts for user."""
        _draw_text_on_image(self.source_image, "Click handle.")
        cv2.imshow(self.window_name, self.source_image)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        while not self.user_input_set():
            cv2.waitKey(1)
        cv2.destroyAllWindows()

    def get_ros_input_handle_and_hinge(self):
        grouped_detections = {}
        for detection in self.detections:
            if detection.results[0].id not in grouped_detections:
                grouped_detections[detection.results[0].id] = []
            grouped_detections[detection.results[0].id].append(detection.bbox)

        average_handle_x = 0.0
        average_handle_y = 0.0
        average_door_x_center = 0.0
        average_door_x_left = 0.0
        average_door_x_right = 0.0

        for door in grouped_detections[0]:
            average_door_x_center += door.center_x
            average_door_x_left += door.center_x - (door.size_x / 2)
            average_door_x_right += door.center_x + (door.size_x / 2)
        average_door_x_center /= len(grouped_detections[0])
        average_door_x_left /= len(grouped_detections[0])
        average_door_x_right /= len(grouped_detections[0])

        for handle in grouped_detections[1]:
            average_handle_x += handle.center_x
            average_handle_y += handle.center_y
        average_handle_x /= len(grouped_detections[1])
        average_handle_y /= len(grouped_detections[1])

        self.handle_position_side_by_side = (average_handle_x, average_handle_y)

        # is the handle to the right or left of the center of the door?
        hinge_x = average_door_x_center
        if average_handle_x < average_door_x_center:
            hinge_x = average_door_x_right
        else:
            hinge_x = average_door_x_left
        self.hinge_position_side_by_side = (hinge_x, average_handle_y)

    def get_walk_to_object_in_image_request(self, debug):
        """Convert from touchpoints in side by side image to a WalkToObjectInImage request.
        Optionally show debug image of touch point.

        Args:
            debug (bool): Show intermediate debug image..

        Returns:
            ManipulationApiRequest: Request with WalkToObjectInImage info populated.
        """

        # Figure out which source the user actually clicked.
        height, width = self.source_image.shape
        assert (
            self.handle_position_side_by_side is not None
        ), "handle position has not yet been found! Cannot continue."
        rotated_pixel = self.handle_position_side_by_side
        if self.mode == "user":
            if self.handle_position_side_by_side[0] > width / 2:
                self.image_detection_source_name = "frontleft_fisheye_image"
                rotated_pixel = (rotated_pixel[0] - width / 2, rotated_pixel[1])
            else:
                self.image_detection_source_name = "frontright_fisheye_image"
        elif self.mode == "ros":
            self.image_detection_source_name = "hand_color_image"

        # Undo pixel rotation by rotation 90 deg CCW.
        manipulation_cmd = WalkToObjectInImage()
        th = -math.pi / 2
        xm = width / 4
        ym = height / 2
        x = rotated_pixel[0] - xm
        y = rotated_pixel[1] - ym
        manipulation_cmd.pixel_xy.x = math.cos(th) * x - math.sin(th) * y + ym
        manipulation_cmd.pixel_xy.y = math.sin(th) * x + math.cos(th) * y + xm

        # Optionally show debug image.
        if debug:
            clicked_cv2 = self.image_dict[self.image_detection_source_name][1]
            c = (255, 0, 0)
            cv2.circle(
                clicked_cv2,
                (int(manipulation_cmd.pixel_xy.x), int(manipulation_cmd.pixel_xy.y)),
                30,
                c,
                5,
            )
            cv2.imshow("Debug", clicked_cv2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Populate the rest of the Manip API request.
        # clicked_image_proto = self.image_dict[self.image_detection_source_name][0]
        clicked_image_proto = self.image_detection_source_name
        manipulation_cmd.frame_name_image_sensor = (
            clicked_image_proto.shot.frame_name_image_sensor
        )
        manipulation_cmd.transforms_snapshot_for_camera.CopyFrom(
            clicked_image_proto.shot.transforms_snapshot
        )
        manipulation_cmd.camera_model.CopyFrom(clicked_image_proto.source.pinhole)
        door_search_dist_meters = 1.25
        manipulation_cmd.offset_distance.value = door_search_dist_meters

        request = ManipulationApiRequest(walk_to_object_in_image=manipulation_cmd)
        return request

    @property
    def vision_tform_sensor(self):
        """Look up vision_tform_sensor for sensor which user clicked.

        Returns:
            math_helpers.SE3Pose
        """
        clicked_image_proto = self.image_dict[self.image_detection_source_name][0]
        frame_name_image_sensor = clicked_image_proto.shot.frame_name_image_sensor
        snapshot = clicked_image_proto.shot.transforms_snapshot
        return frame_helpers.get_a_tform_b(
            snapshot, frame_helpers.VISION_FRAME_NAME, frame_name_image_sensor
        )

    @property
    def hinge_side(self):
        """Calculate if hinge is on left or right side of door based on user touchpoints.

        Returns:
            DoorCommand.HingeSide
        """
        assert self.handle_position_side_by_side is not None
        assert self.hinge_position_side_by_side is not None
        handle_x = self.handle_position_side_by_side[0]
        hinge_x = self.hinge_position_side_by_side[0]
        if handle_x < hinge_x:
            hinge_side = door_pb2.DoorCommand.HINGE_SIDE_RIGHT  # type: ignore
        else:
            hinge_side = door_pb2.DoorCommand.HINGE_SIDE_LEFT  # type: ignore
        return hinge_side


def _draw_text_on_image(image, text):
    font_scale = 4
    thickness = 4
    font = cv2.FONT_HERSHEY_PLAIN  # type: ignore
    (text_width, text_height) = cv2.getTextSize(  # type:ignore
        text, font, fontScale=font_scale, thickness=thickness
    )[0]

    rectangle_bgr = (255, 255, 255)
    text_offset_x = 10
    text_offset_y = image.shape[0] - 25
    border = 10
    box_coords = (
        (text_offset_x - border, text_offset_y + border),
        (text_offset_x + text_width + border, text_offset_y - text_height - border),
    )
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)  # type: ignore
    cv2.putText(  # type: ignore
        image,
        text,
        (text_offset_x, text_offset_y),
        font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=thickness,
    )


def open_door(robot, request_manager, snapshot):
    """Command the robot to automatically open a door via the door service API.

    Args:
        robot: (Robot) Interface to Spot robot.
        request_manager: (RequestManager) Object for bookkeeping user touch points.
        snapshot: (TransformSnapshot) Snapshot from the WalkToObjectInImage command which contains
            the 3D location reported from a raycast based on the user hinge touch point.
    """

    rospy.loginfo("Opening door...")

    # Using the raycast intersection point and the
    vision_tform_raycast = frame_helpers.get_a_tform_b(
        snapshot, frame_helpers.VISION_FRAME_NAME, frame_helpers.RAYCAST_FRAME_NAME
    )
    vision_tform_sensor = request_manager.vision_tform_sensor
    raycast_point_wrt_vision = vision_tform_raycast.get_translation()
    ray_from_camera_to_object = (
        raycast_point_wrt_vision - vision_tform_sensor.get_translation()
    )
    ray_from_camera_to_object_norm = np.sqrt(np.sum(ray_from_camera_to_object ** 2))
    ray_from_camera_normalized = (
        ray_from_camera_to_object / ray_from_camera_to_object_norm
    )

    auto_cmd = door_pb2.DoorCommand.AutoGraspCommand()
    auto_cmd.frame_name = frame_helpers.VISION_FRAME_NAME
    search_dist_meters = 0.25
    search_ray = search_dist_meters * ray_from_camera_normalized
    search_ray_start_in_frame = raycast_point_wrt_vision - search_ray
    auto_cmd.search_ray_start_in_frame.CopyFrom(
        geometry_pb2.Vec3(
            x=search_ray_start_in_frame[0],
            y=search_ray_start_in_frame[1],
            z=search_ray_start_in_frame[2],
        )
    )

    search_ray_end_in_frame = raycast_point_wrt_vision + search_ray
    auto_cmd.search_ray_end_in_frame.CopyFrom(
        geometry_pb2.Vec3(
            x=search_ray_end_in_frame[0],
            y=search_ray_end_in_frame[1],
            z=search_ray_end_in_frame[2],
        )
    )

    auto_cmd.hinge_side = request_manager.hinge_side
    auto_cmd.swing_direction = door_pb2.DoorCommand.SWING_DIRECTION_UNKNOWN

    door_command = door_pb2.DoorCommand.Request(auto_grasp_command=auto_cmd)
    request = door_pb2.OpenDoorCommandRequest(door_command=door_command)

    # Command the robot to open the door.
    door_client = robot.ensure_client(DoorClient.default_service_name)
    response = door_client.open_door(request)

    feedback_request = door_pb2.OpenDoorFeedbackRequest()
    feedback_request.door_command_id = response.door_command_id

    timeout_sec = 60.0
    end_time = time.time() + timeout_sec
    while time.time() < end_time:
        feedback_response = door_client.open_door_feedback(feedback_request)
        if (
            feedback_response.status
            != basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING
        ):
            raise Exception("Door command reported status ")
        if (
            feedback_response.feedback.status
            == door_pb2.DoorCommand.Feedback.STATUS_COMPLETED
        ):
            rospy.loginfo("Opened door.")
            return
        time.sleep(0.5)
    raise Exception("Door command timed out. Try repositioning the robot.")


def pitch_up(spot_wrapper):
    pose_hold_time = 10.0
    euler_x = 0
    euler_y = -1 * math.pi / 6.0
    euler_z = 0

    spot_wrapper.spot_pose(euler_x, euler_y, euler_z, pose_hold_time)


def get_door_handle_and_hinge(door_detection_sub_topic, robot, spot_wrapper):
    door_detections: List[Detection2D] = []

    def cb(msg):
        door_detections.append(msg)

    rospy.loginfo("Waiting for door detections on " + door_detection_sub_topic)
    sub = rospy.Subscriber(door_detection_sub_topic, Detection2D, cb)
    rospy.sleep(1.5)
    if len(door_detections) > 0:
        rospy.loginfo("Received door detection message, collecting more...")
        rospy.sleep(1.5)
        sub.unregister()

        request_manager = RequestManager()
        request_manager.init_ros_image_mode(door_detections)
        request_manager.get_ros_input_handle_and_hinge()

    else:
        rospy.logwarn("Autodetection failed: falling back to human-in-the-loop")
        # Pitch the robot up. This helps ensure that the door is in the field of view of the front
        # cameras.
        pitch_up(spot_wrapper)
        time.sleep(2.0)
        request_manager = default_door_detection_service_proxy(robot)

    # Tell the robot to walk toward the door.
    manipulation_feedback = walk_to_object_in_image(robot, request_manager, debug=False)

    # The ManipulationApiResponse for the WalkToObjectInImage command returns a transform snapshot
    # that contains where user clicked door handle point intersects the world. We use this
    # intersection point to execute the door command.
    snapshot = manipulation_feedback.transforms_snapshot_manipulation_data
    return snapshot, request_manager


def execute_open_door(robot, spot_wrapper, door_detection_sub_topic):
    """High level behavior sequence for commanding the robot to open a door."""

    snapshot, request_manager = get_door_handle_and_hinge(
        door_detection_sub_topic, robot, spot_wrapper
    )

    # Execute the door command.
    open_door(robot, request_manager, snapshot)


def default_door_detection_service_proxy(robot):
    # Capture images from the two from cameras.
    sources = ["frontleft_fisheye_image", "frontright_fisheye_image"]
    image_dict = get_images_as_cv2(robot, sources)

    # Get handle and hinge locations from user input.
    window_name = "Door Detection"
    request_manager = RequestManager()
    request_manager.init_user_input_mode(image_dict, window_name)
    request_manager.get_user_input_handle_and_hinge()
    assert (
        request_manager.user_input_set()
    ), "Failed to get user input for handle and hinge."

    return request_manager


def open_door_main(robot, spot_wrapper, door_detection_sub_topic):
    """Main function for opening door."""
    success: bool
    try:
        # Execute open door command sequence.
        execute_open_door(robot, spot_wrapper, door_detection_sub_topic)
        comment = "Opened door successfully."
        robot.operator_comment(comment)
        rospy.loginfo(comment)
        success = True
    except Exception as e:
        comment = "Failed to open door: {},\n{}".format(e, e.with_traceback)
        rospy.logerr(comment)
        robot.operator_comment(comment)
        rospy.loginfo(comment)
        success = False
    return success
