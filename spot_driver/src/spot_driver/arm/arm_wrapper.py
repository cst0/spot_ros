from bosdyn.api import arm_command_pb2, estop_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client.robot import RobotCommandClient
from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives
import rospy
import actionlib

from std_srvs.srv import Trigger, TriggerResponse
from spot_msgs.msg import OpenDoorAction
from spot_msgs.srv import OpenDoor, SetArmImpedanceParams, SetArmImpedanceParamsResponse
from vision_msgs.msg import Detection2D
from spot_driver.arm.arm_utilities.object_grabber import object_grabber_main
from spot_driver.arm.arm_utilities.door_opener import open_door_main
from control_msgs.msg import FollowJointTrajectoryAction
from actionlib import SimpleActionServer

from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME,
                                         GROUND_PLANE_FRAME_NAME, HAND_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b)
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.robot_state import RobotStateClient
import math
import time
from bosdyn.util import seconds_to_timestamp, seconds_to_duration


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

        self.arm_impedance_parameters = rospy.Service(
            "arm_impedance_parameters",
            SetArmImpedanceParams,
            self.handle_arm_impedance_matrix,
            )

        self.arm_joint_trajectory_server = SimpleActionServer(
            "arm_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
            execute_cb=self.handle_arm_joint_trajectory)
        self.arm_joint_trajectory_server.start()

        self.grasp_point_userinput_srv = rospy.Service(
            "grasp_point_userinput",
            Trigger,
            self.handle_grasp_point_userinput,
        )

        dds = "door_detection_service"
        self.door_detection_service_proxy = None
        if rospy.has_param(dds):
            self.door_detection_service_proxy = rospy.ServiceProxy(
                rospy.get_param(dds), Detection2D
            )
        self.object_detection_service_proxy = None

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
            self._robot, self._spot_wrapper, self.door_detection_service_proxy
        ), "Complete!"

    def handle_grasp_point_userinput(self, _):
        rospy.loginfo("Got grasp point request (w/ user input)")
        return object_grabber_main(
            self._robot, self._spot_wrapper
        ), "Complete!"

    def handle_arm_impedance_matrix(self, params):
        robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Tell the robot to stand up, parameterized to enable the body to adjust its height to
        # assist manipulation. For this demo, that means the robot's base will descend, enabling
        # the hand to reach the ground.
        # The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        ## robot.logger.info("Commanding robot to stand...")
        command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)

        body_control = spot_command_pb2.BodyControlParams(
            body_assist_for_manipulation=spot_command_pb2.BodyControlParams.
            BodyAssistForManipulation(enable_hip_height_assist=True, enable_body_yaw_assist=False))
        # Define a stand command that we'll send with the rest of our arm commands so we keep
        # adjusting the body for the arm
        stand_command = RobotCommandBuilder.synchro_stand_command(
            params=spot_command_pb2.MobilityParams(body_control=body_control))

        # First, let's pick a task frame that is in front of the robot on the ground.
        robot_state = robot_state_client.get_robot_state()
        odom_T_task = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                    ODOM_FRAME_NAME, HAND_FRAME_NAME)

        # Now, let's set our tool frame to be the tip of the robot's bottom jaw. Flip the
        # orientation so that when the hand is pointed downwards, the tool's z-axis is
        # pointed upward.
        wr1_T_tool = SE3Pose(0.23589, 0, -0.03943, Quat.from_pitch(-math.pi / 2))

        # Execute the Cartesian command.
        #### cmd_id = command_client.robot_command(robot_cmd)
        #### block_until_arm_arrives(command_client, cmd_id, 3.0)

        # First, let's do an impedance command where we set all of our stiffnesses high and
        # move around. This will act similar to a position command, but be slightly less stiff.
        robot_cmd = robot_command_pb2.RobotCommand()
        robot_cmd.CopyFrom(stand_command)  # Make sure we keep adjusting the body for the arm
        impedance_cmd = robot_cmd.synchronized_command.arm_command.arm_impedance_command

        # Set up our root frame, task frame, and tool frame.
        impedance_cmd.root_frame_name = ODOM_FRAME_NAME
        impedance_cmd.root_tform_task.CopyFrom(odom_T_task.to_proto())
        impedance_cmd.wrist_tform_tool.CopyFrom(wr1_T_tool.to_proto())

        # Set up stiffness and damping matrices. Note: if these values are set too high,
        # the arm can become unstable. Currently, these are the max stiffness and
        # damping values that can be set.
        impedance_cmd.diagonal_stiffness_matrix.CopyFrom(
            geometry_pb2.Vector(values=[params.linear_stiffness.x, params.linear_stiffness.y, params.linear_stiffness.z,
                                        params.rotational_stiffness.x, params.rotational_stiffness.y, params.rotational_stiffness.z]))
        impedance_cmd.diagonal_damping_matrix.CopyFrom(
            geometry_pb2.Vector(values=[params.linear_damping.x, params.linear_damping.y, params.linear_damping.z,
                                        params.rotational_damping.x, params.rotational_damping.y, params.rotational_damping.z]))
        # Set up our `desired_tool` trajectory. This is where we want the tool to be with respect
        # to the task frame. The stiffness we set will drag the tool towards `desired_tool`.
        traj = impedance_cmd.task_tform_desired_tool
        pt1 = traj.points.add()
        pt1.time_since_reference.CopyFrom(seconds_to_duration(2.0))
        pt1.pose.CopyFrom(SE3Pose(0, 0, 0, Quat.from_pitch(-math.pi / 2)).to_proto())

        # Execute the impedance command.
        cmd_id = command_client.robot_command(robot_cmd)
        time.sleep(2.0)
        return SetArmImpedanceParamsResponse(success=True)

    def handle_arm_joint_trajectory(self, goal):
        joint_names = ['arm0.sh0', 'arm0.sh1', 'arm0.el0', 'arm0.el1', 'arm0.wr0', 'arm0.wr1']
        joint_positions = []
        for name in joint_names:
            if name in goal.trajectory.joint_names:
                joint_positions.append(goal.trajectory.joint_names.index(name))
            else:
                msg = "Unsupported joint name {}. It must be {}".format(name, joint_names)
                rospy.logerr(msg)
                return self.arm_joint_trajectory_server.set_aborted(text=msg)
        command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
        # initialize data
        start_time = time.time()
        ref_time = seconds_to_timestamp(start_time)
        times = []
        positions = []
        velocities = []
        # start sending commands
        for point in goal.trajectory.points:
            print([point.positions[i] for i in joint_positions])
            print([point.velocities[i] for i in joint_positions])
            print(point.time_from_start.to_sec())
            positions.append([point.positions[i] for i in joint_positions])
            velocities.append([point.velocities[i] for i in joint_positions])
            times.append(point.time_from_start.to_sec())
            total_time = point.time_from_start.to_sec()
            print(len(times), len(goal.trajectory.points))
            if len(times) >= 10 or len(times) == len(goal.trajectory.points):
                robot_cmd = RobotCommandBuilder.arm_joint_move_helper(joint_positions=positions,
                                                                      joint_velocities=velocities,
                                                                      times=times, ref_time=ref_time,
                                                                      max_acc=10000, max_vel=10000)
                cmd_id = command_client.robot_command(robot_cmd)
                times = []
                positions = []
                velocities = []
        block_until_arm_arrives(command_client, cmd_id, total_time + 3)  # 3[sec] is buffer
        return self.arm_joint_trajectory_server.set_succeeded()

