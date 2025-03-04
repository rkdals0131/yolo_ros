import cv2
from typing import List, Dict
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from sensor_msgs.msg import Image
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class YoloDebugNode(LifecycleNode):
    
    def __init__(self) -> None:
        super().__init__("yolo_debug_node")
        
        # Declare model parameters with default values
        self.declare_parameter("model", "yolov8n.pt")         # Path to YOLO model file
        self.declare_parameter("device", "cuda:0")            # Device to run inference on (GPU)
        self.declare_parameter("threshold", 0.5)              # Confidence threshold for detections
        self.declare_parameter("iou", 0.5)                    # IoU threshold for NMS
        self.declare_parameter("max_det", 100)                # Maximum number of detections

        # Image size parameters
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 480)

        # ROS parameters
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)  # QoS reliability policy
        
        # Missing attribute definition - should be added
        self.type_to_model = {"yolov8": YOLO}  # Missing attribute - add this
        self.model_type = "yolov8"             # Missing attribute - add this
        self.threshold = 0.5                   # Missing threshold attribute - add this
        self.iou = 0.5                         # Missing IoU attribute - add this
        self.max_det = 100                     # Missing max_det attribute - add this

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Lifecycle callback when configuring the node.
        Sets up the YOLO model and publishers/subscribers.
        """
        # Load model from parameter
        self.model = YOLO(self.get_parameter("model").get_parameter_value().string_value)
        self.device = self.get_parameter("device").get_parameter_value().string_value

        # Get image size from parameters
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value
        
        # Get reliability parameter for image QoS
        self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value
        
        # Set threshold, IoU and max_det from parameters - missing in original code
        self.threshold = self.get_parameter("threshold").get_parameter_value().double_value
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value

        # Configure QoS profile for image subscription
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,)
        
        # Create publisher for detection results
        self._pub = self.create_lifecycle_publisher(DetectionArray, "detections", 10)
        self.cv_bridge = CvBridge()  # Bridge for converting between ROS and OpenCV images

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Lifecycle callback when activating the node.
        Sets up the YOLO model and starts the image subscription.
        """
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            # This should use self.model directly instead of creating a new instance
            # self.yolo = self.type_to_model[self.model_type](self.model)
            self.yolo = self.model  # This is more appropriate since we already loaded the model
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exist")
            return TransitionCallbackReturn.ERROR

        try:
            self.get_logger().info("Trying to fuse model...")
            self.yolo.fuse()  # Fuse model layers for optimization
        except TypeError as e:
            self.get_logger().warn(f"Error while fusing: {e}")

        # Create subscription to image topic
        self._sub = self.create_subscription(
            Image, "image_raw", self.image_cb, self.image_qos_profile
        )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Lifecycle callback when deactivating the node.
        Releases resources associated with model and subscriptions.
        """
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")
        
        # Release model and clean GPU memory
        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        # Clean up subscription
        self.destroy_subscription(self._sub)
        self._sub = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Lifecycle callback for cleaning up resources when shutting down.
        """
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")
        
        # Clean up publisher
        self.destroy_publisher(self._pub)
        self._pub = None

        # Delete QoS profile
        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Lifecycle callback when node is shutting down.
        """
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        
        # Bug: This should call on_shutdown not on_cleanup
        # super().on_cleanup(state)
        super().on_shutdown(state)  # Corrected method call
        
        self.get_logger().info(f"[{self.get_name()}] Shut down")

        return TransitionCallbackReturn.SUCCESS

    def parse_hypothesis(self, results: Results) -> List[Dict]:
        """
        Extract detection hypothesis (class, confidence) from YOLO results.
        
        Args:
            results: Detection results from YOLO model
            
        Returns:
            List of dictionaries containing class ID, name, and confidence score
        """
        hypothesis_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)

        elif results.obb:  # Oriented bounding boxes
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:
        """
        Extract bounding box information from YOLO results.
        
        Args:
            results: Detection results from YOLO model
            
        Returns:
            List of BoundingBox2D messages
        """
        boxes_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                box = BoundingBox2D()
                
                # Bug: This line overwrites the box object with a tensor
                # box = box_data.xywh[0]
                
                # Correct implementation:
                xywh = box_data.xywh[0]  # Get bounding box in xywh format
                box.center.position.x = float(xywh[0])
                box.center.position.y = float(xywh[1])
                box.size.x = float(xywh[2])
                box.size.y = float(xywh[3])

                # append msg
                boxes_list.append(box)

        elif results.obb:  # Oriented bounding boxes
            for i in range(results.obb.cls.shape[0]):
                box = BoundingBox2D()
                
                # Bug: This line overwrites the box object with a tensor
                # box = results.obb.xywhr[i]
                
                # Correct implementation:
                xywhr = results.obb.xywhr[i]  # Get oriented bounding box
                box.center.position.x = float(xywhr[0])
                box.center.position.y = float(xywhr[1])
                box.size.x = float(xywhr[2])
                box.size.y = float(xywhr[3])
                # Note: This ignores rotation (r) in the xywhr tensor
                
                # append msg
                boxes_list.append(box)

        return boxes_list

    def image_cb(self, msg: Image) -> None:
        """
        Callback function for processing incoming images.
        Performs YOLO detection and publishes results.
        
        Args:
            msg: ROS Image message
        """
        try:  # Added try-except block for error handling
            # Convert image + predict
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=self.threshold,
                iou=self.iou,
                imgsz=(self.imgsz_height, self.imgsz_width),
                max_det=self.max_det,
                device=self.device,
            )
            results: Results = results[0].cpu()  # Get first batch result and move to CPU

            hypothesis = []
            boxes = []
            
            if results.boxes or results.obb:
                hypothesis = self.parse_hypothesis(results)
                boxes = self.parse_boxes(results)

            # Create detection msgs
            detections_msg = DetectionArray()

            # Bug: This loop should use the length of hypothesis/boxes, not results
            # for i in range(len(results)):
            for i in range(len(hypothesis)):  # Corrected loop range
                aux_msg = Detection()

                # Simplified conditional logic
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]
                aux_msg.bbox = boxes[i]

                detections_msg.detections.append(aux_msg)

            # Publish detections
            detections_msg.header = msg.header
            self._pub.publish(detections_msg)

            # Clean up resources
            del results
            del cv_image
            
        except Exception as e:  # Added exception handling
            self.get_logger().error(f"Error in image callback: {e}")


def main():
    rclpy.init()                   
    node = YoloDebugNode()         
    node.trigger_configure()       
    node.trigger_activate()        
    rclpy.spin(node)               
    node.destroy_node()            
    rclpy.shutdown()               

if __name__ == "__main__":
    main()