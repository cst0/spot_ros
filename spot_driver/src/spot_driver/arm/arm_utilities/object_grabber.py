# Modified from the original Boston Dynamics example code for more general use.
import time
import rospy

import cv2
import numpy as np

from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME,
    get_vision_tform_body,
    math_helpers,
)
from bosdyn.client.manipulation_api_client import ManipulationApiClient

g_image_click = None
g_image_display = None


def arm_object_grasp(robot, spot_wrapper, options):
    #image_responses = spot_wrapper._image_client.get_image_from_sources(
    #    options["image_source"]
    #)
    #image_responses = spot_wrapper.gripper_images[0]
    image = spot_wrapper.front_images[0]

    #if len(image_responses) != 1:
    #    print("Got invalid number of images: " + str(len(image_responses)))
    #    print(image_responses)
    #    assert False
    #image = image_responses[0]

    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
    else:
        dtype = np.uint8
    img = np.fromstring(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(image.shot.image.rows, image.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Show the image to the user and wait for them to click on a pixel
    robot.logger.info("Click on an object to start grasping...")
    image_title = "Click to grasp"
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, cv_mouse_callback)

    global g_image_click, g_image_display
    g_image_display = img
    cv2.imshow(image_title, g_image_display)
    while g_image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            # Quit
            print('"q" pressed, exiting.')
            exit(0)

    cv2.destroyAllWindows()

    robot.logger.info(
        "Picking object at image location ("
        + str(g_image_click[0])
        + ", "
        + str(g_image_click[1])
        + ")"
    )

    pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

    # Build the proto
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=image.shot.transforms_snapshot,
        frame_name_image_sensor=image.shot.frame_name_image_sensor,
        camera_model=image.source.pinhole,
    )

    # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
    add_grasp_constraint(options, grasp, spot_wrapper._robot_state_client)

    # Ask the robot to pick up the object
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp
    )

    # Send the request
    manipulation_api_client = robot.ensure_client(
        ManipulationApiClient.default_service_name
    )
    cmd_response = manipulation_api_client.manipulation_api_command(
        manipulation_api_request=grasp_request
    )

    # Get feedback from the robot
    while True:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id
        )

        # Send the request
        response = manipulation_api_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request
        )

        rospy.logdebug_throttle_identical(1.0,
            "Current state: "+manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state),
        )

        if (
            response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
            or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
        ):
            break

    rospy.loginfo("Completed grasp!")

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        # print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = "Click to grasp"
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


def add_grasp_constraint(options, grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    use_vector_constraint = (
        options["force_top_down_grasp"] or options["force_horizontal_grasp"]
    )

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if options["force_top_down_grasp"]:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if options["force_horizontal_grasp"]:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper
        )
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo
        )

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif options["force_45_angle_grasp"]:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_Q_grasp.to_proto()
        )

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif options["force_squeeze_grasp"]:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()


def object_grabber_main(
    robot,
    spot_wrapper,
    image_source="frontleft_fisheye_image",
    force_top_down_grasp=False,
    force_horizontal_grasp=False,
    force_45_angle_grasp=False,
    force_squeeze_grasp=False,
):

    options = {
        "image_source": image_source,
        "force_top_down_grasp": force_top_down_grasp,
        "force_horizontal_grasp": force_horizontal_grasp,
        "force_45_angle_grasp": force_45_angle_grasp,
        "force_squeeze_grasp": force_squeeze_grasp,
    }

    num = 0
    if force_top_down_grasp:
        num += 1
    if force_horizontal_grasp:
        num += 1
    if force_45_angle_grasp:
        num += 1
    if force_squeeze_grasp:
        num += 1
    if num > 1:
        print(
            "Error: cannot force more than one type of grasp. Defaulting to top-down."
        )
        options["force_top_down_grasp"] = True
        options["force_horizontal_grasp"] = False
        options["force_45_angle_grasp"] = False
        options["force_squeeze_grasp"] = False

    try:
        arm_object_grasp(robot, spot_wrapper, options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        rospy.logerr(exc)
        return False
