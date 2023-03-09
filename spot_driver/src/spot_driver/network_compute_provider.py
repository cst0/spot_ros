import rospy

from std_msgs.msg import Header
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox, RectArray, Rect, ClassificationResult
from sensor_msgs.msg import CompressedImage, Image

import cv2
import math
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.api import network_compute_bridge_pb2
from bosdyn.api import image_pb2
from google.protobuf import wrappers_pb2
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers

# Copied from https://dev.bostondynamics.com/docs/python/fetch_tutorial/fetch6
kImageSources = [
    'hand_color_image',
    'frontleft_fisheye_image', 'frontright_fisheye_image',
    'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image'
]

def get_bounding_box_image(response):
    dtype = np.uint8
    img = np.fromstring(response.image_response.shot.image.data, dtype=dtype)
    if response.image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(response.image_response.shot.image.rows,
                          response.image_response.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Convert to BGR so we can draw colors
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes in the image for all the detections.
    for obj in response.object_in_image:
        conf_msg = wrappers_pb2.FloatValue()
        obj.additional_properties.Unpack(conf_msg)
        confidence = conf_msg.value

        polygon = []
        min_x = float('inf')
        min_y = float('inf')
        for v in obj.image_properties.coordinates.vertexes:
            polygon.append([v.x, v.y])
            min_x = min(min_x, v.x)
            min_y = min(min_y, v.y)

        polygon = np.array(polygon, np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

        caption = "{} {:.3f}".format(obj.name, confidence)
        cv2.putText(img, caption, (int(min_x), int(min_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img


def find_rectangle_px(polygon):
    """Returns center x/y and width/height"""
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for vert in polygon.vertexes:
        if vert.x < min_x:
            min_x = vert.x
        if vert.y < min_y:
            min_y = vert.y
        if vert.x > max_x:
            max_x = vert.x
        if vert.y > max_y:
            max_y = vert.y
    width = math.fabs(max_x - min_x)
    height = math.fabs(max_y - min_y)
    x = width / 2.0 + min_x
    y = height / 2.0 + min_y
    return (x, y, width, height)


class NetworkComputeProvider():
    """Provide NetworkComupte result to ROS"""

    def __init__(self):
        pass

    def shutdown(self):
        rospy.loginfo("Shutting down NetworkComputeProvider")
        rospy.Rate(0.25).sleep()

    def get_objects(self, image_source, header):
        time_start = rospy.Time.now()
        # Build a network compute request for this image source.
        image_source_and_service = network_compute_bridge_pb2.ImageSourceAndService(
            image_source=image_source)

        # Input data:
        #   model name
        #   minimum confidence (between 0 and 1)
        #   if we should automatically rotate the image
        resp_commands = []
        for model in self.ml_models:
            input_data = network_compute_bridge_pb2.NetworkComputeInputData(
                image_source_and_service=image_source_and_service,
                model_name=model,
                min_confidence=self.ml_confidence,
                rotate_image=network_compute_bridge_pb2.NetworkComputeInputData.
                ROTATE_IMAGE_ALIGN_HORIZONTAL)

            # Server data: the service name
            server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
                service_name=self.ml_service)

            # Pack and send the request.
            process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(
                input_data=input_data, server_config=server_data)

            resp_commands.append(self.network_compute_client.network_compute_bridge_command_async(
                process_img_req))
        rospy.loginfo("Prepare NCB command for {} models: {:.4f}".format(len(self.ml_models), (rospy.Time.now()-time_start).to_sec()))
        ret = []
        for resp_command in resp_commands:
            resp = resp_command.result()
            rospy.loginfo("Receives command results         : {:.4f}".format((rospy.Time.now()-time_start).to_sec()))
            for obj in resp.object_in_image:
                # Get the label
                obj_label = obj.name.split('_label_')[-1]

            img = None
            if self.use_gui or self.pub_image.get_num_connections() > 0 or self.pub_image_comp.get_num_connections() > 0:
                img = get_bounding_box_image(resp)
                image_full = resp.image_response

            if self.use_gui:
                # Show the image
                cv2.imshow("NetworkComputeProvider", img)
                cv2.waitKey(15)

            if (not img is None) and (self.pub_image.get_num_connections() > 0):
                self.pub_image.publish(self.bridge.cv2_to_imgmsg(
                    img, encoding='bgr8'))

            if (not img is None) and (self.pub_image_comp.get_num_connections() > 0):
                msg = CompressedImage()
                msg.header = header
                msg.format = "jpeg"
                msg.data = np.array(cv2.imencode('.jpg', img)[1]).tostring()
                self.pub_image_comp.publish(msg)

            if len(resp.object_in_image) > 0:
                vision_tform_robot = frame_helpers.get_a_tform_b(
                    self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
                    frame_helpers.BODY_FRAME_NAME,
                    frame_helpers.VISION_FRAME_NAME)
                for obj in resp.object_in_image:
                    # Get the label
                    obj_label = obj.name.split('_label_')[-1]
                    conf_msg = wrappers_pb2.FloatValue()
                    obj.additional_properties.Unpack(conf_msg)
                    conf = conf_msg.value

                    try:
                        vision_tform_obj = frame_helpers.get_a_tform_b(
                            obj.transforms_snapshot,
                            frame_helpers.VISION_FRAME_NAME,
                            obj.image_properties.frame_name_image_coordinates)
                    except bosdyn.client.frame_helpers.ValidateFrameTreeError:
                        # No depth data available.
                        rospy.logwarn("{} founds {}, but could not get 3D data".format(source, obj_label))
                        vision_tform_obj = None

                    if vision_tform_obj is not None:
                        ret.append([obj, resp.image_response, vision_tform_robot * vision_tform_obj])
            rospy.loginfo("Publih ROS data                  : {:.4f}".format((rospy.Time.now()-time_start).to_sec()))

        return ret

    def main(self):
        """Main function for the NetworkComputeProvider"""
        rospy.init_node('ncb_provider_ros', anonymous=True)

        self.username = rospy.get_param('~username', 'default_value')
        self.password = rospy.get_param('~password', 'default_value')
        self.hostname = rospy.get_param('~hostname', 'default_value')
        self.image_sources = rospy.get_param('~image_sources', kImageSources)
        self.ml_service = rospy.get_param('~ml_service', 'default_value')
        self.ml_models = rospy.get_param('~ml_models', '[default_value]')
        self.ml_confidence = rospy.get_param('~ml_confidence', 0.6)
        self.use_gui = rospy.get_param('~use_gui', False)

        self.pub_image = rospy.Publisher('~output', Image, queue_size=1)
        self.pub_image_comp = rospy.Publisher('~output/compressed', CompressedImage, queue_size=1)
        self.pub_all_pose = rospy.Publisher('~pose_array', PoseArray, queue_size=1)
        self.pub_all_bbox = rospy.Publisher('~bbox_array', BoundingBoxArray, queue_size=1)
        self.pub_all_result = rospy.Publisher('~class', ClassificationResult, queue_size=1)
        self.pub_rects = {}
        self.pub_class = {}
        self.pub_bbox = {}
        for source in self.image_sources:
            self.pub_rects[source] = rospy.Publisher('~{}/rects'.format(source), RectArray, queue_size=1)
            self.pub_class[source] = rospy.Publisher('~{}/class'.format(source), ClassificationResult, queue_size=1)
            self.pub_bbox[source] = rospy.Publisher('~{}/bbox_array'.format(source), BoundingBoxArray, queue_size=1)

        if self.use_gui:
            cv2.namedWindow("NetworkComputeProvider")
            cv2.waitKey(500)

        rospy.loginfo("Starting NetworkComputeProvider " + str(self.ml_service) + "/" + str(self.ml_models) + " for ROS at "+str(self.hostname)+" as "+str(self.username)+" "+str(self.password))

        sdk = bosdyn.client.create_standard_sdk('NetworkComputeProviderClient')
        sdk.register_service_client(NetworkComputeBridgeClient)
        robot = sdk.create_robot(self.hostname)
        robot.authenticate(self.username, self.password)
        bosdyn.client.util.authenticate(robot)

        # Time sync is necessary so that time-based filter requests can be converted
        robot.time_sync.wait_for_sync()

        self.network_compute_client = robot.ensure_client(
            NetworkComputeBridgeClient.default_service_name)
        rospy.loginfo("Available models : {}".format(
            self.network_compute_client.list_available_models(self.ml_service).available_models))
        available_labels = []
        for available_label in map(lambda x: x.available_labels, filter(lambda x: x.model_name in self.ml_models, list(self.network_compute_client.list_available_models(self.ml_service).labels))):
            available_labels.extend(available_label)
        
        self.robot_state_client = robot.ensure_client(
            RobotStateClient.default_service_name)
        
        while not rospy.is_shutdown():
            try:
                #check if ml_confidence is updated
                ml_confidence = rospy.get_param('~ml_confidence', 0.6)
                if not ml_confidence == self.ml_confidence:
                    rospy.logwarn("ml_confidence is updated {} <-- {}".format(ml_confidence, self.ml_confidence))
                    self.ml_confidence = ml_confidence

                # Capture an image and run ML on it
                header = Header(stamp = rospy.Time.now())
                all_label_names = []
                all_label_proba = []
                all_label_poses = []
                all_label_sizes = []
                all_bounding_boxes = []
                for source in self.image_sources:
                    objects = self.get_objects(image_source=source, header=header)

                    if len(objects) == 0:
                        continue
                    
                    header.stamp = rospy.Time(objects[0][0].acquisition_time.seconds,
                                              objects[0][0].acquisition_time.nanos)
                    header.frame_id = source
                    rect_msg = RectArray(header=header)

                    label_names = []
                    label_proba = []
                    label_poses = []
                    label_sizes = []
                    bounding_boxes = []
                    for obj, image, vision_tform_target in objects:
                        label_names.append(obj.name.split('_label_')[-1])
                        conf_msg = wrappers_pb2.FloatValue()
                        obj.additional_properties.Unpack(conf_msg)
                        label_proba.append(conf_msg.value)
                        label_poses.append(Pose(
                                            position = Point(x = vision_tform_target.x,
                                                             y = vision_tform_target.y,
                                                             z = vision_tform_target.z),
                                            orientation = Quaternion(x = vision_tform_target.rot.x,
                                                                     y = vision_tform_target.rot.y,
                                                                     z = vision_tform_target.rot.z,
                                                                     w = vision_tform_target.rot.w)))
                        label_sizes.append(Vector3(x = obj.bounding_box_properties.size_ewrt_frame.y,
                                                   y = obj.bounding_box_properties.size_ewrt_frame.y,
                                                   z = obj.bounding_box_properties.size_ewrt_frame.z))
                        (x, y, width, height) = find_rectangle_px(obj.image_properties.coordinates)
                        rect_msg.rects.append(Rect(x = int(x - width/2), y = int(y - height/2), width = int(width), height = int(height)))
                        bounding_boxes.append(BoundingBox(header = header,
                                                          label = available_labels.index(label_names[-1]),
                                                          value = label_proba[-1],
                                                          pose = label_poses[-1],
                                                          dimensions = label_sizes[-1]))

                    rospy.loginfo("{} founds {}".format(source, list(zip(label_names, label_proba))))
                    
                    # publish classifier rectangle
                    self.pub_rects[source].publish(rect_msg)

                    # publish classifier results
                    cls_msg = ClassificationResult(
                        header=header,
                        classifier='self.ml_model',
                        labels=[available_labels.index(name) for name in label_names],
                        target_names=available_labels,
                        label_names=label_names,
                        label_proba=label_proba)
                    self.pub_class[source].publish(cls_msg)

                    # publish box array
                    self.pub_bbox[source].publish(BoundingBoxArray(header=header, boxes=bounding_boxes))

                    all_label_names.extend(label_names)
                    all_label_proba.extend(label_proba)
                    all_label_poses.extend(label_poses)
                    all_label_sizes.extend(label_sizes)
                    all_bounding_boxes.extend(bounding_boxes)
                    
                # publish bouding box
                if len(all_label_names) == 0:
                    continue

                header.frame_id = 'body'
                self.pub_all_pose.publish(PoseArray(header=header, poses=all_label_poses))

                self.pub_all_bbox.publish(BoundingBoxArray(header=header, boxes=all_bounding_boxes))
                
                self.pub_all_result.publish(
                    ClassificationResult(
                        header = header,
                        classifier = 'self.ml_model',
                        labels = [available_labels.index(name) for name in all_label_names],
                        target_names = available_labels,
                        label_names = all_label_names,
                        label_proba = all_label_proba))

            except Exception as e:
                rospy.logerr('Error:{}'.format(e))
                pass
