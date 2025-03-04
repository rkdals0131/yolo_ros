import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')

        self.declare_parameter('input_mode', 'ros2') # 'webcam' 또는 'ros2'
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('webcam_id', 0)

        self.input_mode = self.get_parameter('input_mode').value
        self.image_topic = self.get_parameter('image_topic').value
        self.webcam_id = self.get_parameter('webcam_id').value

        # 퍼블리셔 생성 (큐 사이즈 10)
        self.publisher_ = self.create_publisher(String, 'cone_info', 10)

        self.cv_bridge = CvBridge()

        # YOLOv8 모델 로드
        self.model = YOLO('/home/user1/yolov12/pretrained_models/yolov8_cone.pt')

        # 입력 모드에 따른 초기화
        if self.input_mode == 'webcam':
            self.setup_webcam()
            # 타이머 생성 (약 30 FPS)
            self.timer = self.create_timer(1/30, self.detect_from_webcam)
        elif self.input_mode == 'ros2':
            # ROS2 이미지 토픽 구독
            self.subscription = self.create_subscription(
                Image,
                self.image_topic,
                self.detect_from_ros2,
                10)
            self.get_logger().info(f"ROS2 이미지 토픽 '{self.image_topic}' 구독 중")
        else:
            self.get_logger().error(f"알 수 없는 입력 모드: {self.input_mode}")
            exit()
    
        # YOLO 클래스 매핑 (모델 학습 시 클래스 순서에 맞게 수정)
        self.class_names = {0: "Blue Cone", 1: "Crimson Cone", 2: "Yellow Cone"}

        # 바운딩박스 색상 매핑 (BGR 형식)
        self.color_mapping = {
            "Crimson Cone": (0, 0, 255),   # 빨강
            "Yellow Cone":  (0, 255, 255), # 노랑
            "Blue Cone":    (255, 0, 0),   # 파랑
            "Unknown":      (0, 255, 0)    # 기본 초록색
        }

        # HSV 색상 범위 정의
        # 빨간색 (Hue는 0-179 스케일에서 정의해야 함)
        # 빨간색은 Hue 범위가 양 끝에 걸쳐 있으므로 두 범위를 정의
        self.lower_crimson_hsv1 = np.array([0, 100, 100])    # 첫 번째 빨간색 범위 (약 0도)
        self.upper_crimson_hsv1 = np.array([20, 255, 255])  
        self.lower_crimson_hsv2 = np.array([170, 100, 100])  # 두 번째 빨간색 범위 (약 340-360도)
        self.upper_crimson_hsv2 = np.array([180, 255, 255])
        
        # 노란색 (약 60도)
        self.lower_yellow_hsv = np.array([21, 165, 200])
        self.upper_yellow_hsv = np.array([33, 255, 255])
        
        # 파란색 (약 240도)
        self.lower_blue_hsv = np.array([100, 100, 70])
        self.upper_blue_hsv = np.array([130, 255, 255])

        # 색상 검증 임계값 (ROI 면적 대비 해당 색상 픽셀 비율)
        self.threshold_ratio = 0.3

    def setup_webcam(self):
        """웹캠 초기화 메소드"""
        self.cap = cv2.VideoCapture(self.webcam_id)
        if not self.cap.isOpened():
            self.get_logger().error("카메라를 열 수 없습니다!")
            exit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.get_logger().info(f"웹캠 ID {self.webcam_id} 초기화 완료")
    
    def detect_from_webcam(self):
        """웹캠 이미지에서 콘 감지"""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("프레임을 읽을 수 없습니다!")
            return
        
        # 공통 처리 함수 호출
        self.process_frame(frame)
    
    def detect_from_ros2(self, msg):
        """ROS2 이미지 토픽에서 콘 감지"""
        try:
            # ROS 이미지 메시지를 OpenCV 이미지로 변환
            frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 공통 처리 함수 호출
            self.process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"이미지 변환 또는 처리 중 오류 발생: {e}")

    def verify_color_hsv(self, roi, color_name):
        """HSV 색 공간을 사용하여 특정 색상 검증"""
        # BGR에서 HSV로 변환
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 색상별 마스크 생성
        if color_name == "Crimson Cone":
            # 빨간색은 두 개의 HSV 범위를 사용하여 마스크 생성 후 합침
            mask1 = cv2.inRange(roi_hsv, self.lower_crimson_hsv1, self.upper_crimson_hsv1)
            mask2 = cv2.inRange(roi_hsv, self.lower_crimson_hsv2, self.upper_crimson_hsv2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name == "Yellow Cone":
            mask = cv2.inRange(roi_hsv, self.lower_yellow_hsv, self.upper_yellow_hsv)
        elif color_name == "Blue Cone":
            mask = cv2.inRange(roi_hsv, self.lower_blue_hsv, self.upper_blue_hsv)
        else:
            return 0.0  # 알 수 없는 색상
        
        # 색상 픽셀 비율 계산
        roi_area = roi.shape[0] * roi.shape[1]
        if roi_area == 0:
            return 0.0
        color_ratio = cv2.countNonZero(mask) / roi_area
        
        return color_ratio, mask

    def process_frame(self, frame):
        results = self.model(frame)
        detection_info = []  # 퍼블리시할 탐지 정보 리스트

        # YOLO 탐지 결과 처리
        for box in results[0].boxes:
            # 바운딩박스 좌표, 클래스, 신뢰도 추출
            x1, y1, x2, y2 = [int(i.item()) for i in box.xyxy[0]]
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # 기본 YOLO 라벨 사용
            label_yolo = self.class_names.get(cls, f"Class {cls}")
            final_label = label_yolo

            # 중심 좌표 계산
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 바운딩박스 영역(ROI) 추출
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # HSV 색 공간에서 색상 검증
            crimson_ratio, crimson_mask = self.verify_color_hsv(roi, "Crimson Cone")
            yellow_ratio, yellow_mask = self.verify_color_hsv(roi, "Yellow Cone")
            blue_ratio, blue_mask = self.verify_color_hsv(roi, "Blue Cone")
            
            # 디버깅용: 마스크 시각화 (선택적)
            # cv2.imshow("Crimson Mask", crimson_mask)
            # cv2.imshow("Yellow Mask", yellow_mask)
            # cv2.imshow("Blue Mask", blue_mask)
            
            # 각 비율을 비교하여 가장 높은 값으로 라벨 결정 (임계값 초과 시)
            max_ratio = max(crimson_ratio, yellow_ratio, blue_ratio)
            if max_ratio > self.threshold_ratio:
                if max_ratio == crimson_ratio:
                    final_label = "Crimson Cone"
                elif max_ratio == yellow_ratio:
                    final_label = "Yellow Cone"
                elif max_ratio == blue_ratio:
                    final_label = "Blue Cone"

            # 최종 라벨에 따른 바운딩박스 색상 선택
            box_color = self.color_mapping.get(final_label, (0, 255, 0))

            # 바운딩박스 및 라벨 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{final_label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # 탐지 정보 문자열 추가 (예: "Crimson Cone: (320, 180)")
            detection_info.append(f"{final_label}: ({cx}, {cy})")

        # 탐지 정보가 있을 경우 ROS 메시지로 퍼블리시
        # *** TODO ****
        # 헤더, 타임스탬프, 이미지상 콘 타입과 갯수? 그리고 픽셀좌표 배열로.
        # stride 정보 주고 1차원 배열로? 어떻게 보낼지 생각해야함. 
        if detection_info:
            msg = String()
            msg.data = "; ".join(detection_info)
            self.publisher_.publish(msg)

        # 결과 프레임 화면에 표시
        cv2.imshow("Cone Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            rclpy.shutdown()

    def set_hsv_ranges(self, color, lower, upper):
        """사용자가 HSV 범위를 동적으로 설정할 수 있는 메소드"""
        if color == "crimson1":
            self.lower_crimson_hsv1 = np.array(lower)
            self.upper_crimson_hsv1 = np.array(upper)
        elif color == "crimson2":
            self.lower_crimson_hsv2 = np.array(lower)
            self.upper_crimson_hsv2 = np.array(upper)
        elif color == "yellow":
            self.lower_yellow_hsv = np.array(lower)
            self.upper_yellow_hsv = np.array(upper)
        elif color == "blue":
            self.lower_blue_hsv = np.array(lower)
            self.upper_blue_hsv = np.array(upper)
        else:
            self.get_logger().error(f"알 수 없는 색상: {color}")

def main(args=None):
    rclpy.init(args=args)
    node = ConeDetector()
    rclpy.spin(node)
    
    # 웹캠 모드인 경우 자원 해제
    if node.input_mode == 'webcam':
        node.cap.release()
    
    cv2.destroyAllWindows()
    node.destroy_node()

if __name__ == '__main__':
    main()