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
        
        # 파라미터 선언
        self.declare_parameter('input_mode', 'ros2')  # 'webcam' 또는 'ros2'
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('webcam_id', 2)
        # 추가: Crimson 검증 활성화 여부 파라미터
        self.declare_parameter('enable_crimson_validation', False)
        
        # 파라미터 읽기
        self.input_mode = self.get_parameter('input_mode').value
        self.image_topic = self.get_parameter('image_topic').value
        self.webcam_id = self.get_parameter('webcam_id').value
        self.enable_crimson_validation = self.get_parameter('enable_crimson_validation').value
        
        # 퍼블리셔 생성 (큐 사이즈 10)
        self.publisher_ = self.create_publisher(String, 'cone_info', 10)
        
        # OpenCV와 ROS2 이미지 변환을 위한 브릿지
        self.cv_bridge = CvBridge()
        
        # YOLOv8 모델 로드
        # 대회장 콘 사진 학습, 블루, 옐로, 레드콘 클래스. 
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
        
        # Crimson 검증을 위한 HSV 및 BGR 범위 설정
        self.lower_crimson1 = np.array([0, 20, 20])
        self.upper_crimson1 = np.array([20, 255, 255])
        self.lower_crimson2 = np.array([160, 20, 20])
        self.upper_crimson2 = np.array([180, 255, 255])
        # 추가로 BGR 기반 범위 (실제 추출한 값 기반)
        self.lower_crimson_bgr = np.array([0, 20, 150])
        self.upper_crimson_bgr = np.array([50, 60, 255])
        
        # Crimson 검증 임계값 (ROI 면적 대비 crimson 픽셀 비율)
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
    
    def process_frame(self, frame):
        """프레임 처리 및 콘 감지 공통 함수"""
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
            
            # OpenCV 기반 Crimson 검증 (모듈화)
            if self.enable_crimson_validation:
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask_crimson_hsv = cv2.bitwise_or(
                    cv2.inRange(roi_hsv, self.lower_crimson1, self.upper_crimson1),
                    cv2.inRange(roi_hsv, self.lower_crimson2, self.upper_crimson2)
                )
                mask_crimson_bgr = cv2.inRange(roi, self.lower_crimson_bgr, self.upper_crimson_bgr)
                mask_crimson = cv2.bitwise_or(mask_crimson_hsv, mask_crimson_bgr)
                
                roi_area = roi.shape[0] * roi.shape[1]
                crimson_ratio = cv2.countNonZero(mask_crimson) / roi_area
                
                if crimson_ratio > self.threshold_ratio:
                    final_label = "Crimson Cone"
            
            # 최종 라벨에 따른 바운딩박스 색상 선택
            box_color = self.color_mapping.get(final_label, (0, 255, 0))
            
            # 바운딩박스 및 라벨 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{final_label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
            # 탐지 정보 문자열 추가 (예: "Crimson Cone: (320, 180)")
            detection_info.append(f"{final_label}: ({cx}, {cy})")
        
        # 탐지 정보가 있을 경우 ROS 메시지로 퍼블리시
        if detection_info:
            msg = String()
            msg.data = "; ".join(detection_info)
            self.publisher_.publish(msg)
        
        # 결과 프레임 화면에 표시
        cv2.imshow("Cone Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.destroy_node()
            rclpy.shutdown()

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
