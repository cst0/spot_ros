from bosdyn.client.robot import RobotCommandClient
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
import rospy
import actionlib

from std_srvs.srv import Trigger, TriggerResponse
from spot_msgs.msg import OpenDoorAction
from spot_driver.arm.arm_utilities.object_grabber import object_grabber_main
from spot_driver.arm.arm_utilities.door_opener import open_door_main

from bosdyn.client.manipulation_api_client import ManipulationApiClient


class ArmWrapper:
    def __init__(self, robot, wrapper, logger):
        self._logger = logger
        self._spot_wrapper = wrapper

        self._robot = robot
        assert (
            self._robot.has_arm()
        ), "You've tried using the arm on your Spot, but no arm was detected!"

        self.open_door_srv = rospy.Service(
            "open_door",
            Trigger,
            self.handle_open_door,
        )

        self.stow_arm_srv = rospy.Service(
            "stow_arm",
            Trigger,
            self.handle_stow_arm,
        )

        self.stow_arm_srv = rospy.Service(
            "unstow_arm",
            Trigger,
            self.handle_unstow_arm,
        )

        self.open_gripper_srv = rospy.Service(
            "gripper_open",
            Trigger,
            self.handle_gripper_open,
        )

        self.open_gripper_srv = rospy.Service(
            "gripper_close",
            Trigger,
            self.handle_gripper_close,
        )

        self.grasp_point_userinput_srv = rospy.Service(
            "grasp_point_userinput",
            Trigger,
            self.handle_grasp_point_userinput,
        )

        self.door_detection_sub_topic = "/door_detections"
        self.object_detection_sub_topic = "/object_detections"

        self._init_bosdyn_clients()
        self._init_actionservers()

    def _init_bosdyn_clients(self):
        self._manip_client = self._robot.ensure_client(
            ManipulationApiClient.default_service_name
        )

    def _init_actionservers(self):
        self.open_door_as = actionlib.SimpleActionServer(
            "open_door",
            OpenDoorAction,
            execute_cb=self.handle_open_door,
            auto_start=False,
        )
        self.open_door_as.start()

    def _send_arm_cmd(self, cmd):
        command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name
        )
        cmd_id = command_client.robot_command(cmd)
        return TriggerResponse(success=block_until_arm_arrives(command_client, cmd_id, 3.0), message="")

    def handle_stow_arm(self, _):
        return self._send_arm_cmd(RobotCommandBuilder.arm_stow_command())

    def handle_unstow_arm(self, _):
        return self._send_arm_cmd(cmd=RobotCommandBuilder.arm_ready_command())

    def handle_gripper_open(self, _):
        return self._send_arm_cmd(RobotCommandBuilder.claw_gripper_open_command())

    def handle_gripper_close(self, _):
        return self._send_arm_cmd(RobotCommandBuilder.claw_gripper_close_command())

    def handle_open_door(self, _):
        rospy.loginfo("Got a open door request")
        return open_door_main(
            self._robot, self._spot_wrapper, self.door_detection_sub_topic
        ), "Complete!"

    def handle_grasp_point_userinput(self, _):
        rospy.loginfo("Got grasp point request (w/ user input)")
        return object_grabber_main(
            self._robot, self._spot_wrapper, self.object_detection_sub_topic
        ), "Complete!"
