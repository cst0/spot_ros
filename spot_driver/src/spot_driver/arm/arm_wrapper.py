from bosdyn.client.robot import RobotCommandClient
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
import rospy
import actionlib

from std_srvs.srv import Trigger
from spot_msgs.msg import OpenDoorAction
from spot_msgs.srv import OpenDoor
from vision_msgs.msg import Detection2D
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

        dds = 'door_detection_service'
        self.door_detection_service_proxy = None
        if rospy.has_param(dds):
            self.door_detection_service_proxy = rospy.ServiceProxy(rospy.get_param(dds), Detection2D)

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

    def handle_stow_arm(self, goal):
        del goal
        command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
        cmd = RobotCommandBuilder.arm_stow_command()
        cmd_id = command_client.robot_command(cmd)
        block_until_arm_arrives(command_client, cmd_id, 3.0)

    def handle_open_door(self, goal):
        del goal
        rospy.loginfo("Got a open door request")
        open_door_main(self._robot, self._spot_wrapper, self.door_detection_service_proxy)
