import rospy
import cv2
import time

from bosdyn.api import basic_command_pb2
from bosdyn.api.spot import door_pb2
from bosdyn.api import geometry_pb2

from bosdyn.api import manipulation_api_pb2
from bosdyn.api.manipulation_api_pb2 import (
    WalkToObjectInImage,
    ManipulationApiRequest,
    ManipulationApiFeedbackRequest,
)
from bosdyn.client.manipulation_api_client import ManipulationApiClient


def walk_to_object_in_image(robot, request_manager, debug):
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
    robot.logger.info("Walking toward door...")
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


def _draw_text_on_image(image, text):
    font_scale = 4
    thickness = 4
    font = cv2.FONT_HERSHEY_PLAIN
    (text_width, text_height) = cv2.getTextSize(
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
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(
        image,
        text,
        (text_offset_x, text_offset_y),
        font,
        fontScale=font_scale,
        color=(0, 0, 0),
        thickness=thickness,
    )


class RequestManager:
    """Helper object for displaying side by side images to the user and requesting user selected
    touchpoints. This class handles the bookkeeping for converting between a touchpoints of side by
    side display image of the frontleft and frontright fisheye images and the individual images.

    Args:
        image_dict: (dict) Dictionary from image source name to (image proto, CV2 image) pairs.
        window_name: (str) Name of display window..
    """

    def __init__(self, image_dict, window_name):
        self.image_dict = image_dict
        self.window_name = window_name
        self.handle_position_side_by_side = None
        self.hinge_position_side_by_side = None
        self._side_by_side = None
        self.clicked_source = None

    def reset(self):
        self.handle_position_side_by_side = None
        self.hinge_position_side_by_side = None
        self._side_by_side = None
        self.clicked_source = None

    @property
    def side_by_side(self):
        """cv2.Image: Side by side rotated frontleft and frontright fisheye images"""
        if self._side_by_side is not None:
            return self._side_by_side

        # Convert PIL images to numpy for processing.
        fr_fisheye_image = self.image_dict["frontright_fisheye_image"][1]
        fl_fisheye_image = self.image_dict["frontleft_fisheye_image"][1]

        # Rotate the images to align with robot Z axis.
        fr_fisheye_image = cv2.rotate(fr_fisheye_image, cv2.ROTATE_90_CLOCKWISE)
        fl_fisheye_image = cv2.rotate(fl_fisheye_image, cv2.ROTATE_90_CLOCKWISE)

        self._side_by_side = np.hstack([fr_fisheye_image, fl_fisheye_image])

        return self._side_by_side

    def user_input_set(self):
        """bool: True if handle and hinge position set."""
        return self.handle_position_side_by_side and self.hinge_position_side_by_side

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.handle_position_side_by_side:
                cv2.circle(self.side_by_side, (x, y), 30, (255, 0, 0), 5)
                _draw_text_on_image(self.side_by_side, "Click hinge.")
                cv2.imshow(self.window_name, self.side_by_side)
                self.handle_position_side_by_side = (x, y)
            elif not self.hinge_position_side_by_side:
                self.hinge_position_side_by_side = (x, y)

    def get_user_input_handle_and_hinge(self):
        """Open window showing the side by side fisheye images with on screen prompts for user."""
        _draw_text_on_image(self.side_by_side, "Click handle.")
        cv2.imshow(self.window_name, self.side_by_side)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        while not self.user_input_set():
            cv2.waitKey(1)
        cv2.destroyAllWindows()

    def get_ros_input_handle_and_hinge(self):
        # check if the door detection ROS service is available
        srv_topic = "/door_detection"
        try:
            rospy.wait_for_service(srv_topic, timeout=1)
        except:
            rospy.logerr("Door detection service failed")
            return False

        try:
            from door_detector_ros.srv import DoorDetection, DoorDetectionResponse
            dd:DoorDetectionResponse = rospy.ServiceProxy(srv_topic, DoorDetection).call()
            self.handle_position_side_by_side = (dd.handle.x, dd.handle.y)
            self.hinge_position_side_by_side = (dd.hinge.x, dd.hinge.y)
        except:
            rospy.logerr("Service call failed")
            return False
        return True


    def get_walk_to_object_in_image_request(self, debug):
        """Convert from touchpoints in side by side image to a WalkToObjectInImage request.
        Optionally show debug image of touch point.

        Args:
            debug (bool): Show intermediate debug image..

        Returns:
            ManipulationApiRequest: Request with WalkToObjectInImage info populated.
        """

        # Figure out which source the user actually clicked.
        height, width = self.side_by_side.shape
        assert self.handle_position_side_by_side is not None
        if self.handle_position_side_by_side[0] > width / 2:
            self.clicked_source = "frontleft_fisheye_image"
            rotated_pixel = self.handle_position_side_by_side
            rotated_pixel = (rotated_pixel[0] - width / 2, rotated_pixel[1])
        else:
            self.clicked_source = "frontright_fisheye_image"
            rotated_pixel = self.handle_position_side_by_side

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
            clicked_cv2 = self.image_dict[self.clicked_source][1]
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
        clicked_image_proto = self.image_dict[self.clicked_source][0]
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
        clicked_image_proto = self.image_dict[self.clicked_source][0]
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
            hinge_side = door_pb2.DoorCommand.HINGE_SIDE_RIGHT
        else:
            hinge_side = door_pb2.DoorCommand.HINGE_SIDE_LEFT
        return hinge_side


class GraspProvider:
    """
    This is the demo grasp provider. It assumes human-in-the-loop in the same
    way the Spot demo examples do, but you can replicate this interface and
    substitute your own model/CV pipeline/etc.
    """

    def __init__(self):
        self.service = rospy.Service(
            "grasp_provider", FindGrasp, self.srv_grasp_provider
        )

    def srv_grasp_provider(self, req):
        sources = ["frontleft_fisheye_image", "frontright_fisheye_image"]
        image_dict = get_images_as_cv2(robot, sources)

        # Get handle and hinge locations from user input.
        window_name = "Open Door Example"
        request_manager = RequestManager(image_dict, window_name)
        # attempt ROS input
        success = request_manager.get_ros_input_handle_and_hinge()
        if not success:
            request_manager.get_user_input_handle_and_hinge()
        assert (
            request_manager.user_input_set()
        ), "Failed to get user input for handle and hinge."
