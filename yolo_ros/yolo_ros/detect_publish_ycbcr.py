import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

class ConeDetector(Node):
    def __init__(self):
        super().__init__('cone_detector')
        # 퍼블리셔 생성 (큐 사이즈 10)
        self.publisher_ = self.create_publisher(String, 'cone_info', 10)

        # YOLOv8 모델 로드
        self.model = YOLO('/home/user1/yolov12/pretrained_models/yolov8_cone.pt')

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("카메라를 열 수 없습니다!")
            exit()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # YOLO 클래스 매핑 (모델 학습 시 클래스 순서에 맞게 수정)
        self.class_names = {0: "Blue Cone", 1: "Crimson Cone", 2: "Yellow Cone"}

        # 바운딩박스 색상 매핑 (BGR 형식)
        self.color_mapping = {
            "Crimson Cone": (0, 0, 255),   # 빨강
            "Yellow Cone":  (0, 255, 255), # 노랑
            "Blue Cone":    (255, 0, 0),   # 파랑
            "Unknown":      (0, 255, 0)    # 기본 초록색
        }

        # 개선된 YCbCr 범위 (재정렬: [Y, Cb, Cr])
        # Crimson Cone: 빨강의 경우, 순수 빨강(BGR: [0,0,255])은 약 [76,85,255]가 나오므로,
        # Cb: 약 60~110, Cr: 220 이상을 요구
        self.lower_crimson_ycbcr = np.array([56, 96, 194])
        self.upper_crimson_ycbcr = np.array([117, 90, 226])
        
        # Yellow Cone: 순수 노랑(BGR: [0,255,255])은 재정렬 후 약 [226, 0, 149]가 나오므로,
        # Cb: 매우 낮은 값(0~20), Cr: 130~170 범위를 사용
        self.lower_yellow_ycbcr = np.array([0, 0, 130])
        self.upper_yellow_ycbcr = np.array([255, 20, 170])
        
        # Blue Cone: 기존 범위를 사용 (필요 시 추가 튜닝 가능)
        self.lower_blue_ycbcr = np.array([0, 150, 85])
        self.upper_blue_ycbcr = np.array([255, 255, 125])

        # 색상 검증 임계값 (ROI 면적 대비 해당 색상 픽셀 비율)
        self.threshold_ratio = 0.25

        # 타이머 생성 (약 30 FPS)
        self.timer = self.create_timer(1/30, self.detect_and_publish)

    def verify_color(self, roi, color_name):
        """YCbCr 색 공간을 사용하여 특정 색상 검증"""
        # BGR에서 YCbCr로 변환 (cv2.COLOR_BGR2YCrCb는 (Y, Cr, Cb) 순서를 반환)
        roi_ycbcr = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(roi_ycbcr)
        roi_ycbcr_reordered = cv2.merge([y, cb, cr])  # 재정렬: [Y, Cb, Cr]
        
        # 색상별 마스크 생성
        if color_name == "Crimson Cone":
            mask = cv2.inRange(roi_ycbcr_reordered, self.lower_crimson_ycbcr, self.upper_crimson_ycbcr)
        elif color_name == "Yellow Cone":
            mask = cv2.inRange(roi_ycbcr_reordered, self.lower_yellow_ycbcr, self.upper_yellow_ycbcr)
        elif color_name == "Blue Cone":
            mask = cv2.inRange(roi_ycbcr_reordered, self.lower_blue_ycbcr, self.upper_blue_ycbcr)
        else:
            return 0.0  # 알 수 없는 색상
        
        # 색상 픽셀 비율 계산
        roi_area = roi.shape[0] * roi.shape[1]
        if roi_area == 0:
            return 0.0
        color_ratio = cv2.countNonZero(mask) / roi_area
        
        return color_ratio

    def detect_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("프레임을 읽을 수 없습니다!")
            return

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

            # YCbCr 색 공간에서 색상 검증
            crimson_ratio = self.verify_color(roi, "Crimson Cone")
            yellow_ratio = self.verify_color(roi, "Yellow Cone")
            blue_ratio = self.verify_color(roi, "Blue Cone")
            
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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
