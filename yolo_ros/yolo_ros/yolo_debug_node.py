import cv2
from typing import List, Dict
from cv_bridge import CvBridge
import numpy as np

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
from std_msgs.msg import String
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray


class YoloDebugNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_debug_node")

        # Declare model parameters with default values
        self.declare_parameter("device", "cuda:0")            # Device to run inference on (GPU)
        self.declare_parameter("threshold", 0.5)              # Confidence threshold for detections
        self.declare_parameter("iou", 0.5)                    # IoU threshold for NMS
        self.declare_parameter("max_det", 100)                # Maximum number of detections

        # Image size parameters
        self.declare_parameter("imgsz_height", 360)
        self.declare_parameter("imgsz_width", 640)

        # ROS parameters
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)  # QoS reliability policy

        # Input mode parameters (new)
        self.declare_parameter('input_mode', 'ros2')          # 'webcam' or 'ros2'
        self.declare_parameter('image_topic', '/image_raw')   # ROS2 image topic
        self.declare_parameter('webcam_id', 2)                # Webcam device ID

        # Set default parameter values
        self.declare_parameter('verification_mode', 'hybrid') # Options: 'hybrid', 'hsv', 'yolo'
        self.declare_parameter('confidence_weight_yolo', 0.5)  # Weight for YOLO confidence (for hybrid)
        self.declare_parameter('confidence_weight_hsv', 0.5)   # Weight for HSV confidence (for hybrid)
        self.declare_parameter('threshold_ratio', 0.25)         # Color verification threshold (for hsv/hybrid)

        # for model parameters
        self.threshold = 0.4
        self.iou = 0.4
        self.max_det = 100

        # Add encoding debugging support
        self.debug_encoding = True
        self.encoding_stats = {
            'original': {},     # Original image encoding statistics
            'successful': {},   # Successfully converted format statistics
            'failed': {}        # Failed conversion format statistics
        }

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """
        Lifecycle callback when configuring the node.
        Sets up the YOLO model and publishers/subscribers.
        """
        # Load model from parameter
        self.model = YOLO('/home/user1/yolov12/pretrained_models/yolov8_cone.pt') # Ensure this path is correct!
        self.device = self.get_parameter("device").get_parameter_value().string_value

        # Get image size from parameters
        self.imgsz_height = self.get_parameter("imgsz_height").get_parameter_value().integer_value
        self.imgsz_width = self.get_parameter("imgsz_width").get_parameter_value().integer_value

        # Get reliability parameter for image QoS
        self.reliability = self.get_parameter("image_reliability").get_parameter_value().integer_value

        # Get input mode parameters (new)
        self.input_mode = self.get_parameter('input_mode').value
        self.image_topic = self.get_parameter('image_topic').value
        self.webcam_id = self.get_parameter('webcam_id').value

        # Set verification parameters
        self.verification_mode = self.get_parameter('verification_mode').value
        self.confidence_weight_yolo = self.get_parameter('confidence_weight_yolo').value
        self.confidence_weight_hsv = self.get_parameter('confidence_weight_hsv').value
        self.threshold_ratio = self.get_parameter('threshold_ratio').value
        # for model parameters
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
        # Add debug image publisher
        self._dbg_pub = self.create_lifecycle_publisher(Image, "dbg_image", 10)
        # Add cone info string publisher (new)
        self._info_pub = self.create_lifecycle_publisher(String, "cone_info", 10)

        self.cv_bridge = CvBridge()  # Bridge for converting between ROS and OpenCV images

        # Setup HSV color ranges for cone classification (new)
        # Crimson (red) cones - defined in two ranges because hue wraps around
        self.lower_crimson_hsv1 = np.array([0, 100, 100])
        self.upper_crimson_hsv1 = np.array([20, 255, 255])
        self.lower_crimson_hsv2 = np.array([170, 100, 100])
        self.upper_crimson_hsv2 = np.array([180, 255, 255])

        # Yellow cones
        self.lower_yellow_hsv = np.array([21, 165, 200])
        self.upper_yellow_hsv = np.array([33, 255, 255])

        # Blue cones
        self.lower_blue_hsv = np.array([100, 100, 70])
        self.upper_blue_hsv = np.array([130, 255, 255])

        # Color name mapping
        self.class_names = {0: "Blue Cone", 1: "Crimson Cone", 2: "Yellow Cone"}

        # Color mapping for visualization (BGR format)
        self.color_mapping = {
            "Crimson Cone": (0, 0, 255),   # Red
            "Yellow Cone":  (0, 255, 255), # Yellow
            "Blue Cone":    (255, 0, 0),   # Blue
            "Unknown":      (0, 255, 0)    # Green (default)
        }

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
            self.yolo = self.model
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exist")
            return TransitionCallbackReturn.ERROR

        try:
            self.get_logger().info("Trying to fuse model...")
            self.yolo.fuse()  # Fuse model layers for optimization
        except TypeError as e:
            self.get_logger().warn(f"Error while fusing: {e}")

        # Setup input based on mode (new)
        if self.input_mode == 'webcam':
            self.setup_webcam()
            # Create timer for webcam processing (~30 FPS)
            self.timer = self.create_timer(1/30, self.process_webcam)
            self.get_logger().info(f"Using webcam ID {self.webcam_id}")
        elif self.input_mode == 'ros2':
            # Create subscription to image topic
            self._sub = self.create_subscription(
                Image, self.image_topic, self.image_cb, self.image_qos_profile
            )
            self.get_logger().info(f"Subscribing to ROS2 image topic: {self.image_topic}")
        else:
            self.get_logger().error(f"Unknown input mode: {self.input_mode}")
            return TransitionCallbackReturn.ERROR

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

        # Clean up based on input mode (new)
        if self.input_mode == 'webcam':
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            self.destroy_timer(self.timer)
            self.timer = None
        elif self.input_mode == 'ros2':
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

        # Clean up publishers
        self.destroy_publisher(self._pub)
        self._pub = None
        self.destroy_publisher(self._dbg_pub)
        self._dbg_pub = None
        self.destroy_publisher(self._info_pub)  # New publisher cleanup
        self._info_pub = None

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

        super().on_shutdown(state)

        self.get_logger().info(f"[{self.get_name()}] Shut down")

        return TransitionCallbackReturn.SUCCESS

    def setup_webcam(self):
        """Initialize webcam capture device"""
        self.cap = cv2.VideoCapture(self.webcam_id)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            return

        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.imgsz_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imgsz_height)
        self.get_logger().info(f"Webcam initialized with resolution {self.imgsz_width}x{self.imgsz_height}")

    def process_webcam(self):
        """Process frames from webcam when in webcam mode"""
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.get_logger().error("Camera not available")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to read frame from camera")
            return
        
        if self.debug_encoding:
            # Log webcam image format (OpenCV uses BGR by default)
            height, width, channels = frame.shape
            encoding = f"BGR ({channels} channels)"
            
            # Update statistics
            if encoding not in self.encoding_stats['original']:
                self.encoding_stats['original'][encoding] = 0
            self.encoding_stats['original'][encoding] += 1
            
            # Log statistics periodically (every 100 frames)
            if sum(self.encoding_stats['original'].values()) % 100 == 0:
                self.log_encoding_stats()

        # Create a dummy header for the webcam image
        header = rclpy.time.Time().to_msg()

        # Process the frame
        self.process_frame(frame, header)

    def verify_color_hsv(self, roi, color_name):
        """
        Verify color in HSV space

        Args:
            roi: Region of interest (cropped image)
            color_name: Name of color to verify

        Returns:
            color_ratio: Ratio of pixels matching the color
            mask: Binary mask of matching pixels
        """
        # Convert BGR to HSV
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Create mask based on color name
        if color_name == "Crimson Cone":
            # Red color spans two ranges in HSV
            mask1 = cv2.inRange(roi_hsv, self.lower_crimson_hsv1, self.upper_crimson_hsv1)
            mask2 = cv2.inRange(roi_hsv, self.lower_crimson_hsv2, self.upper_crimson_hsv2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name == "Yellow Cone":
            mask = cv2.inRange(roi_hsv, self.lower_yellow_hsv, self.upper_yellow_hsv)
        elif color_name == "Blue Cone":
            mask = cv2.inRange(roi_hsv, self.lower_blue_hsv, self.upper_blue_hsv)
        else:
            return 0.0, None

        # Calculate ratio of color pixels
        roi_area = roi.shape[0] * roi.shape[1]
        if roi_area == 0:
            return 0.0, None

        color_ratio = cv2.countNonZero(mask) / roi_area

        return color_ratio, mask

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

                xywhr = results.obb.xywhr[i]  # Get oriented bounding box
                box.center.position.x = float(xywhr[0])
                box.center.position.y = float(xywhr[1])
                box.size.x = float(xywhr[2])
                box.size.y = float(xywhr[3])

                # append msg
                boxes_list.append(box)

        return boxes_list

    def draw_detections(self, cv_image: np.ndarray, results: Results,
                        verified_labels: List[str] = None) -> np.ndarray:
        """
        Draw detection results on the image with color verification.

        Args:
            cv_image: OpenCV image (in BGR format)
            results: YOLO detection results
            verified_labels: List of verified color labels

        Returns:
            Image with drawn detections
        """
        # Create a copy of the image to draw on
        debug_image = cv_image.copy()

        # Draw bounding boxes for detected objects
        if results.boxes:
            for i, box in enumerate(results.boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class information
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Use verified labels if available
                if verified_labels and i < len(verified_labels):
                    cls_name = verified_labels[i]
                else:
                    cls_name = self.yolo.names[cls_id]

                # Get color from color mapping
                color = self.color_mapping.get(cls_name, (0, 255, 0))  # Default: green

                # Draw rectangle
                cv2.rectangle(debug_image, (x1, y1), (x2, y2), color, 2)

                # Add label text
                label = f"{cls_name} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(debug_image, (x1, y1), (x1 + t_size[0], y1 - t_size[1] - 10), color, -1)
                cv2.putText(debug_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Handle oriented bounding boxes if present (You can add OBB drawing logic here if needed)
        # if hasattr(results, 'obb') and results.obb is not None:
        #     for i in range(results.obb.cls.shape[0]):
        #         # Use verified label if available
        #         if verified_labels and i < len(verified_labels):
        #             cls_name = verified_labels[i]
        #         else:
        #             cls_id = int(results.obb.cls[i])
        #             cls_name = self.yolo.names[cls_id]

        #         # Get color
        #         color = self.color_mapping.get(cls_name, (0, 255, 0))

        #         # Get oriented box points and draw
        #         # ... (OBB drawing code - to be implemented if needed)

        return debug_image

    def process_frame(self, cv_image, header):
        """
        Process a frame (from webcam or ROS topic)

        Args:
            cv_image: OpenCV image
            header: Message header for timestamps
        """
        try:
            # Verify input image is valid
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("Received empty image for processing")
                return

            # 입력 이미지가 BGR 형식임을 가정하고 RGB로 변환하여 YOLO에 전달
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Run YOLO detection
            results = self.yolo.predict(
                source=rgb_image,
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
            verified_labels = []
            detection_info = []

            if results.boxes or results.obb:
                hypothesis = self.parse_hypothesis(results)
                boxes = self.parse_boxes(results)

                # HSV 색상 검증
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = cv_image[y1:y2, x1:x2]  # Extract Region of Interest (ROI)
                    cls_id = int(box.cls[0])
                    yolo_label = self.yolo.names[cls_id]

                    crimson_ratio, _ = self.verify_color_hsv(roi, "Crimson Cone")
                    yellow_ratio, _ = self.verify_color_hsv(roi, "Yellow Cone")
                    blue_ratio, _ = self.verify_color_hsv(roi, "Blue Cone")

                    max_ratio = max(crimson_ratio, yellow_ratio, blue_ratio)
                    hsv_color_label = "Unknown"  # Default HSV label

                    if max_ratio > self.threshold_ratio:
                        if max_ratio == crimson_ratio:
                            hsv_color_label = "Crimson Cone"
                        elif max_ratio == yellow_ratio:
                            hsv_color_label = "Yellow Cone"
                        elif max_ratio == blue_ratio:
                            hsv_color_label = "Blue Cone"

                    if self.verification_mode == 'yolo':
                        final_label = yolo_label  # Pure YOLO label
                    elif self.verification_mode == 'hsv':
                        final_label = hsv_color_label # Pure HSV label
                    elif self.verification_mode == 'hybrid':
                        # Hybrid verification: Combine YOLO and HSV
                        yolo_conf = float(box.conf[0])
                        hsv_conf = max_ratio  # Use max_ratio as HSV confidence

                        # Weighted average confidence (example - you can adjust weights)
                        combined_conf = (yolo_conf * self.confidence_weight_yolo) + (hsv_conf * self.confidence_weight_hsv)

                        if combined_conf > self.threshold: # Use overall threshold for combined confidence
                            if hsv_color_label != "Unknown": # If HSV is somewhat confident in a color
                                final_label = hsv_color_label # Favor HSV color if confident enough
                            else:
                                final_label = yolo_label # Fallback to YOLO if HSV is unsure
                        else:
                            final_label = "Unknown" # If combined confidence is too low

                        # Alternative Hybrid Logic (Prioritize HSV if strong color match)
                        # if max_ratio > self.threshold_ratio:
                        #     final_label = hsv_color_label # If HSV strong, use HSV label
                        # else:
                        #     final_label = yolo_label      # Otherwise, use YOLO label


                    verified_labels.append(final_label)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    detection_info.append(f"{final_label}: ({cx}, {cy})")

            # Create detection array message
            detections_msg = DetectionArray()
            detections_msg.header.stamp = header.stamp if hasattr(header, 'stamp') else header
            detections_msg.header.frame_id = header.frame_id if hasattr(header, 'frame_id') else "camera"

            # Create individual detection messages
            for i in range(len(hypothesis)):
                aux_msg = Detection()
                aux_msg.class_id = hypothesis[i]["class_id"]

                # Use color-verified label if available
                if i < len(verified_labels):
                    aux_msg.class_name = verified_labels[i]
                else:
                    aux_msg.class_name = hypothesis[i]["class_name"]

                aux_msg.score = hypothesis[i]["score"]  # You might want to adjust score in hybrid mode
                aux_msg.bbox = boxes[i]
                detections_msg.detections.append(aux_msg)

            # Publish detections
            self._pub.publish(detections_msg)

            # Publish cone info string
            if detection_info:
                info_msg = String()
                info_msg.data = "; ".join(detection_info)
                self._info_pub.publish(info_msg)

            # Draw and publish debug image
            debug_image = self.draw_detections(cv_image, results, verified_labels)
            debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            debug_msg.header.stamp = header.stamp if hasattr(header, 'stamp') else header
            debug_msg.header.frame_id = header.frame_id if hasattr(header, 'frame_id') else "camera"
            self._dbg_pub.publish(debug_msg)

            # Clean up resources
            del results

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def log_encoding_stats(self):
        """Log encoding statistics"""
        stats_msg = "\n=== Image Encoding Statistics ===\n"
        
        # Original encoding statistics
        stats_msg += "Original image encodings:\n"
        for enc, count in self.encoding_stats['original'].items():
            stats_msg += f"  - {enc}: {count} times\n"
        
        # Successful conversion statistics
        stats_msg += "\nSuccessful conversion formats:\n"
        for enc, count in self.encoding_stats['successful'].items():
            stats_msg += f"  - {enc}: {count} times\n"
        
        # Failed conversion statistics
        stats_msg += "\nFailed conversion formats:\n"
        for enc, count in self.encoding_stats['failed'].items():
            stats_msg += f"  - {enc}: {count} times\n"
        
        # Recommended encoding format
        stats_msg += "\nRecommended encoding format: "
        
        # Find format with highest success rate
        successful_encodings = self.encoding_stats['successful']
        if successful_encodings:
            best_encoding = max(successful_encodings.items(), key=lambda x: x[1])[0]
            stats_msg += f"{best_encoding}\n"
        else:
            stats_msg += "Not enough data yet.\n"
        
        stats_msg += "=======================\n"
        self.get_logger().info(stats_msg)

    def image_cb(self, msg: Image) -> None:
        """
        Callback function for processing incoming ROS images.

        Args:
            msg: ROS Image message
        """
        try:
            # Check message itself
            if msg is None or not msg.data:
                self.get_logger().warn("Received empty image message")
                return

            # Use passthrough conversion to preserve original encoding
            try:
                # Convert using passthrough to maintain original format
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'passthrough')
                
                # Convert to BGR based on source encoding
                if msg.encoding.lower() in ['rgb8', 'rgb16']:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                elif msg.encoding.lower() == 'rgba8':
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
                elif msg.encoding.lower() == 'mono8' and len(cv_image.shape) == 2:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                # BGR8 is already in the correct format, no conversion needed
                
            except Exception as e:
                self.get_logger().error(f"Image conversion failed: {e}")
                return
            
            # Verify image is valid
            if cv_image is None or cv_image.size == 0:
                self.get_logger().warn("Empty frame after conversion")
                return
            
            # Process the frame
            self.process_frame(cv_image, msg.header)

        except Exception as e:
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