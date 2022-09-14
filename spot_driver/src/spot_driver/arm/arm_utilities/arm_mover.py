# Based on code originally from the Boston Dynamics example documentation.
# Modified for more general use by cst0 (chris thierauf, chris@cthierauf.com)
from __future__ import print_function
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    block_until_arm_arrives,
)


def arm_absolute(spot_wrapper, x, y, z, qx=0, qy=0, qz=0, qw=1, seconds=2):
    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

    flat_body_T_hand = geometry_pb2.SE3Pose(
        position=hand_ewrt_flat_body, rotation=flat_body_Q_hand
    )

    robot_state = spot_wrapper._robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        ODOM_FRAME_NAME,
        GRAV_ALIGNED_BODY_FRAME_NAME,
    )

    assert odom_T_flat_body is not None
    odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.x,
        odom_T_hand.y,
        odom_T_hand.z,
        odom_T_hand.rot.w,  # type: ignore
        odom_T_hand.rot.x,  # type: ignore
        odom_T_hand.rot.y,  # type: ignore
        odom_T_hand.rot.z,  # type: ignore
        ODOM_FRAME_NAME,
        seconds,
    )

    # Send the request
    cmd_id = spot_wrapper._robot_command_client.robot_command(arm_command)
    return block_until_arm_arrives(spot_wrapper._robot_command_client, cmd_id)


def arm_relative(spot_wrapper, x=0, y=0, z=0, qx=0, qy=0, qz=0, qw=1, seconds=2):
    # get current spot arm position
    robot_state = spot_wrapper._robot_state_client.get_robot_state()
    odom_T_hand = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot,
        ODOM_FRAME_NAME,
        "hand",
    )
    assert odom_T_hand is not None

    # send current position plus relative postion to arm_absolute
    return arm_absolute(
        spot_wrapper,
        odom_T_hand.x + x,
        odom_T_hand.y + y,
        odom_T_hand.z + z,
        odom_T_hand.rot.w + qw,  # type: ignore
        odom_T_hand.rot.x + qx,  # type: ignore
        odom_T_hand.rot.y + qy,  # type: ignore
        odom_T_hand.rot.z + qz,  # type: ignore
        seconds,
    )
