import rospy
import actionlib
from sensor_msgs.msg import Image, CameraInfo, JointState
from tf2_msgs.msg import TFMessage
from spot_msgs.msg import Metrics
from spot_msgs.msg import LeaseArray
from spot_msgs.msg import FootStateArray
from spot_msgs.msg import EStopStateArray
from spot_msgs.msg import WiFiState
from spot_msgs.msg import PowerState
from spot_msgs.msg import BehaviorFaultState
from spot_msgs.msg import SystemFaultState
from spot_msgs.msg import BatteryStateArray
from spot_msgs.msg import Feedback
from spot_msgs.msg import MobilityParams
from spot_msgs.msg import NavigateToAction
from spot_msgs.msg import TrajectoryAction
from spot_msgs.srv import ListGraph, ListGraphResponse
from spot_msgs.srv import SetLocomotion, SetLocomotionResponse
from spot_msgs.srv import ClearBehaviorFault, ClearBehaviorFaultResponse
from spot_msgs.srv import SetVelocity, SetVelocityResponse
from spot_msgs.srv import SpotPose, SpotPoseResponse
from spot_msgs.srv import Dock, DockResponse, GetDockState, GetDockStateResponse
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistWithCovarianceStamped, Twist, Pose
from nav_msgs.msg import Odometry


class SpotROSMock:
    def __init__(self, use_arm=True):
        self.use_arm = use_arm

    def main(self):
        rospy.init_node('spot_driver', anonymous=True)

        self.back_image_pub = rospy.Publisher("camera/back/image", Image, queue_size=10)
        self.frontleft_image_pub = rospy.Publisher(
            "camera/frontleft/image", Image, queue_size=10
        )
        self.frontright_image_pub = rospy.Publisher(
            "camera/frontright/image", Image, queue_size=10
        )
        self.left_image_pub = rospy.Publisher("camera/left/image", Image, queue_size=10)
        self.right_image_pub = rospy.Publisher(
            "camera/right/image", Image, queue_size=10
        )

        # Depth #
        self.back_depth_pub = rospy.Publisher("depth/back/image", Image, queue_size=10)
        self.frontleft_depth_pub = rospy.Publisher(
            "depth/frontleft/image", Image, queue_size=10
        )
        self.frontright_depth_pub = rospy.Publisher(
            "depth/frontright/image", Image, queue_size=10
        )
        self.left_depth_pub = rospy.Publisher("depth/left/image", Image, queue_size=10)
        self.right_depth_pub = rospy.Publisher(
            "depth/right/image", Image, queue_size=10
        )

        # Image Camera Info #
        self.back_image_info_pub = rospy.Publisher(
            "camera/back/camera_info", CameraInfo, queue_size=10
        )
        self.frontleft_image_info_pub = rospy.Publisher(
            "camera/frontleft/camera_info", CameraInfo, queue_size=10
        )
        self.frontright_image_info_pub = rospy.Publisher(
            "camera/frontright/camera_info", CameraInfo, queue_size=10
        )
        self.left_image_info_pub = rospy.Publisher(
            "camera/left/camera_info", CameraInfo, queue_size=10
        )
        self.right_image_info_pub = rospy.Publisher(
            "camera/right/camera_info", CameraInfo, queue_size=10
        )
        # Depth Camera Info #
        self.back_depth_info_pub = rospy.Publisher(
            "depth/back/camera_info", CameraInfo, queue_size=10
        )
        self.frontleft_depth_info_pub = rospy.Publisher(
            "depth/frontleft/camera_info", CameraInfo, queue_size=10
        )
        self.frontright_depth_info_pub = rospy.Publisher(
            "depth/frontright/camera_info", CameraInfo, queue_size=10
        )
        self.left_depth_info_pub = rospy.Publisher(
            "depth/left/camera_info", CameraInfo, queue_size=10
        )
        self.right_depth_info_pub = rospy.Publisher(
            "depth/right/camera_info", CameraInfo, queue_size=10
        )

        self.gripper_image_pubs = []
        self.gripper_camera_info_pubs = []
        if self.use_arm:
            for t in ["hand_color", "hand_depth", "hand_image"]:
                self.gripper_image_pubs.append(
                    rospy.Publisher("camera/" + t + "/image", Image, queue_size=10)
                )
                self.gripper_camera_info_pubs.append(
                    rospy.Publisher(
                        "camera/" + t + "/camera_info", CameraInfo, queue_size=10
                    )
                )

        # Status Publishers #
        self.joint_state_pub = rospy.Publisher(
            "joint_states", JointState, queue_size=10
        )
        """Defining a TF publisher manually because of conflicts between Python3 and tf"""
        self.tf_pub = rospy.Publisher("tf", TFMessage, queue_size=10)
        self.metrics_pub = rospy.Publisher("status/metrics", Metrics, queue_size=10)
        self.lease_pub = rospy.Publisher("status/leases", LeaseArray, queue_size=10)
        self.odom_twist_pub = rospy.Publisher(
            "odometry/twist", TwistWithCovarianceStamped, queue_size=10
        )
        self.odom_pub = rospy.Publisher("odometry", Odometry, queue_size=10)
        self.feet_pub = rospy.Publisher("status/feet", FootStateArray, queue_size=10)
        self.estop_pub = rospy.Publisher("status/estop", EStopStateArray, queue_size=10)
        self.wifi_pub = rospy.Publisher("status/wifi", WiFiState, queue_size=10)
        self.power_pub = rospy.Publisher(
            "status/power_state", PowerState, queue_size=10
        )
        self.battery_pub = rospy.Publisher(
            "status/battery_states", BatteryStateArray, queue_size=10
        )
        self.behavior_faults_pub = rospy.Publisher(
            "status/behavior_faults", BehaviorFaultState, queue_size=10
        )
        self.system_faults_pub = rospy.Publisher(
            "status/system_faults", SystemFaultState, queue_size=10
        )

        self.feedback_pub = rospy.Publisher("status/feedback", Feedback, queue_size=10)

        self.mobility_params_pub = rospy.Publisher(
            "status/mobility_params", MobilityParams, queue_size=10
        )

        rospy.Subscriber("cmd_vel", Twist, self.cmdVelCallback, queue_size=1)
        rospy.Subscriber("body_pose", Pose, self.bodyPoseCallback, queue_size=1)

        rospy.Service("claim", Trigger, self.handle_claim)
        rospy.Service("force_claim", Trigger, self.handle_force_claim)
        rospy.Service("release", Trigger, self.handle_release)
        rospy.Service("stop", Trigger, self.handle_stop)
        rospy.Service("self_right", Trigger, self.handle_self_right)
        rospy.Service("sit", Trigger, self.handle_sit)
        rospy.Service("stand", Trigger, self.handle_stand)
        rospy.Service("power_on", Trigger, self.handle_power_on)
        rospy.Service("power_off", Trigger, self.handle_safe_power_off)

        rospy.Service("estop/hard", Trigger, self.handle_estop_hard)
        rospy.Service("estop/gentle", Trigger, self.handle_estop_soft)
        rospy.Service("estop/release", Trigger, self.handle_estop_disengage)

        rospy.Service("spot_pose", SpotPose, self.handle_spot_pose)
        rospy.Service("stair_mode", SetBool, self.handle_stair_mode)
        rospy.Service("locomotion_mode", SetLocomotion, self.handle_locomotion_mode)
        rospy.Service("max_velocity", SetVelocity, self.handle_max_vel)
        rospy.Service(
            "clear_behavior_fault",
            ClearBehaviorFault,
            self.handle_clear_behavior_fault,
        )

        rospy.Service("list_graph", ListGraph, self.handle_list_graph)

        # Docking
        rospy.Service("dock", Dock, self.handle_dock)
        rospy.Service("undock", Trigger, self.handle_undock)
        rospy.Service("docking_state", GetDockState, self.handle_get_docking_state)

        self.navigate_as = actionlib.SimpleActionServer(
            "navigate_to",
            NavigateToAction,
            execute_cb=self.handle_navigate_to,
            auto_start=False,
        )
        self.navigate_as.start()

        self.trajectory_server = actionlib.SimpleActionServer(
            "trajectory",
            TrajectoryAction,
            execute_cb=self.handle_trajectory,
            auto_start=False,
        )
        self.trajectory_server.start()

        if self.use_arm:
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

        rospy.spin()

    def handle_grasp_point_userinput(self, _):
        return TriggerResponse(success=True, message="")

    def handle_gripper_open(self, _):
        return TriggerResponse(success=True, message="")

    def handle_gripper_close(self, _):
        return TriggerResponse(success=True, message="")

    def handle_stow_arm(self, _):
        return TriggerResponse(success=True, message="")

    def handle_unstow_arm(self, _):
        return TriggerResponse(success=True, message="")

    def handle_open_door(self, _):
        return TriggerResponse(success=True, message="")

    def handle_list_graph(self, _):
        return ListGraphResponse(graphs=[])

    def handle_clear_behavior_fault(self, _):
        return ClearBehaviorFaultResponse(success=True)

    def handle_max_vel(self, _):
        return SetVelocityResponse(success=True)

    def handle_locomotion_mode(self, _):
        return SetLocomotionResponse(success=True)

    def handle_stair_mode(self, _):
        return SetBoolResponse(success=True)

    def handle_spot_pose(self, _):
        return SpotPoseResponse(success=True)

    def handle_estop_hard(self, _):
        return TriggerResponse(success=True)

    def handle_estop_soft(self, _):
        return TriggerResponse(success=True)

    def handle_estop_disengage(self, _):
        return TriggerResponse(success=True)

    def handle_power_on(self, _):
        return TriggerResponse(success=True)

    def handle_safe_power_off(self, _):
        return TriggerResponse(success=True)

    def handle_sit(self, _):
        return TriggerResponse(success=True)

    def handle_stand(self, _):
        return TriggerResponse(success=True)

    def handle_self_right(self, _):
        return TriggerResponse(success=True)

    def handle_stop(self, _):
        return TriggerResponse(success=True)

    def handle_release(self, _):
        return TriggerResponse(success=True)

    def handle_force_claim(self, _):
        return TriggerResponse(success=True)

    def handle_claim(self, _):
        return TriggerResponse(success=True)

    def handle_dock(self, _):
        return DockResponse(success=True)

    def handle_undock(self, _):
        return TriggerResponse(success=True)

    def handle_get_docking_state(self, _):
        return GetDockStateResponse(success=True)

    def handle_navigate_to(self, _):
        self.navigate_as.set_succeeded()

    def handle_trajectory(self, _):
        self.trajectory_server.set_succeeded()

    def cmdVelCallback(self, _):
        pass

    def jointStateCallback(self, _):
        pass

    def bodyPoseCallback(self, _):
        pass
