# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.duration import Duration
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray

# DebugNode 클래스: YOLO 객체 감지 결과를 시각화하는 디버그 노드
# 이 노드는 LifecycleNode를 상속받아 ROS2 라이프사이클 관리 기능을 제공합니다.
# 주요 기능:
# - 감지된 객체에 바운딩 박스 그리기
# - 객체 마스크 시각화
# - 키포인트(관절 등) 시각화
# - 3D 바운딩 박스 및 키포인트를 RViz에서 볼 수 있는 마커로 변환
class DebugNode(LifecycleNode):

    def __init__(self) -> None:
        # 노드 이름을 "debug_node"로 초기화
        super().__init__("debug_node")

        # 클래스별 색상을 저장할 딕셔너리 초기화
        self._class_to_color = {}
        # OpenCV와 ROS 이미지 간 변환을 위한 브릿지 초기화
        self.cv_bridge = CvBridge()

        # 파라미터 선언: 이미지 QoS 신뢰성 설정
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)

    # 노드 구성 단계 (configure 상태)
    # 파라미터 로드, 퍼블리셔 생성 등의 초기화 작업 수행
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # 이미지 QoS 프로필 설정
        self.image_qos_profile = QoSProfile(
            reliability=self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        # 퍼블리셔 생성:
        # - 디버그 이미지 (시각화된 결과)
        # - 바운딩 박스 마커 (RViz용)
        # - 키포인트 마커 (RViz용)
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._bb_markers_pub = self.create_publisher(MarkerArray, "dgb_bb_markers", 10)
        self._kp_markers_pub = self.create_publisher(MarkerArray, "dgb_kp_markers", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    # 노드 활성화 단계 (activate 상태)
    # 구독자 생성 및 콜백 등록
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        # 구독자 생성:
        # - 원본 이미지
        # - 감지 결과
        self.image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=self.image_qos_profile
        )
        self.detections_sub = message_filters.Subscriber(
            self, DetectionArray, "detections", qos_profile=10
        )

        # 이미지와 감지 결과를 시간 동기화하여 처리하기 위한 동기화 설정
        # 0.5초 내의 메시지를 동기화하며, 최대 10개 메시지를 버퍼링
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.image_sub, self.detections_sub), 10, 0.5
        )
        self._synchronizer.registerCallback(self.detections_cb)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    # 노드 비활성화 단계 (deactivate 상태)
    # 구독자 및 동기화 객체 정리
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        # 구독자 정리
        self.destroy_subscription(self.image_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)

        # 동기화 객체 정리
        del self._synchronizer

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    # 노드 정리 단계 (cleanup 상태)
    # 퍼블리셔 정리
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        # 퍼블리셔 정리
        self.destroy_publisher(self._dbg_pub)
        self.destroy_publisher(self._bb_markers_pub)
        self.destroy_publisher(self._kp_markers_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    # 노드 종료 단계 (shutdown 상태)
    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    # 바운딩 박스 그리기 메서드
    # 감지된 객체의 바운딩 박스를 이미지에 그림
    def draw_box(
        self,
        cv_image: np.ndarray,  # 그릴 이미지
        detection: Detection,   # 감지 결과
        color: Tuple[int],      # 그릴 색상
    ) -> np.ndarray:

        # 감지 정보 추출
        class_name = detection.class_name  # 클래스 이름 (예: 'person', 'car')
        score = detection.score            # 감지 신뢰도 점수
        box_msg: BoundingBox2D = detection.bbox  # 바운딩 박스 정보
        track_id = detection.id            # 추적 ID (있는 경우)

        # 바운딩 박스의 좌상단, 우하단 좌표 계산
        min_pt = (
            round(box_msg.center.position.x - box_msg.size.x / 2.0),
            round(box_msg.center.position.y - box_msg.size.y / 2.0),
        )
        max_pt = (
            round(box_msg.center.position.x + box_msg.size.x / 2.0),
            round(box_msg.center.position.y + box_msg.size.y / 2.0),
        )

        # 회전된 사각형의 네 꼭지점 정의
        rect_pts = np.array(
            [
                [min_pt[0], min_pt[1]],
                [max_pt[0], min_pt[1]],
                [max_pt[0], max_pt[1]],
                [min_pt[0], max_pt[1]],
            ]
        )

        # 회전 행렬 계산 (바운딩 박스가 회전된 경우)
        rotation_matrix = cv2.getRotationMatrix2D(
            (box_msg.center.position.x, box_msg.center.position.y),
            -np.rad2deg(box_msg.center.theta),
            1.0,
        )

        # 사각형의 꼭지점을 회전
        rect_pts = np.int0(cv2.transform(np.array([rect_pts]), rotation_matrix)[0])

        # 회전된 사각형 그리기
        for i in range(4):
            pt1 = tuple(rect_pts[i])
            pt2 = tuple(rect_pts[(i + 1) % 4])
            cv2.line(cv_image, pt1, pt2, color, 2)

        # 텍스트 작성 (클래스 이름, 추적 ID, 신뢰도 점수)
        label = f"{class_name}"
        label += f" ({track_id})" if track_id else ""
        label += " ({:.3f})".format(score)
        pos = (min_pt[0] + 5, min_pt[1] + 25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, label, pos, font, 1, color, 1, cv2.LINE_AA)

        return cv_image

    # 마스크 그리기 메서드
    # 감지된 객체의 마스크(세그멘테이션)를 이미지에 그림
    def draw_mask(
        self,
        cv_image: np.ndarray,  # 그릴 이미지
        detection: Detection,   # 감지 결과
        color: Tuple[int],      # 그릴 색상
    ) -> np.ndarray:

        # 마스크 정보 추출
        mask_msg = detection.mask
        # 마스크 포인트를 numpy 배열로 변환
        mask_array = np.array([[int(ele.x), int(ele.y)] for ele in mask_msg.data])

        # 마스크 데이터가 있는 경우에만 처리
        if mask_msg.data:
            # 원본 이미지 복사
            layer = cv_image.copy()
            # 마스크 영역을 색상으로 채움
            layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
            # 원본 이미지와 마스크 레이어를 블렌딩 (40% 원본, 60% 마스크)
            cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
            # 마스크 윤곽선 그리기
            cv_image = cv2.polylines(
                cv_image,
                [mask_array],
                isClosed=True,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        return cv_image

    # 키포인트 그리기 메서드
    # 감지된 객체의 키포인트(관절 등)를 이미지에 그림
    def draw_keypoints(self, cv_image: np.ndarray, detection: Detection) -> np.ndarray:

        # 키포인트 정보 추출
        keypoints_msg = detection.keypoints

        # Ultralytics 어노테이터 초기화 (키포인트 색상 및 스켈레톤 정보 사용)
        ann = Annotator(cv_image)

        # 각 키포인트 그리기
        kp: KeyPoint2D
        for kp in keypoints_msg.data:
            # 키포인트 색상 결정 (17개 키포인트인 경우 COCO 포맷으로 간주)
            color_k = (
                [int(x) for x in ann.kpt_color[kp.id - 1]]
                if len(keypoints_msg.data) == 17
                else colors(kp.id - 1)
            )

            # 키포인트 원 그리기
            cv2.circle(
                cv_image,
                (int(kp.point.x), int(kp.point.y)),
                5,
                color_k,
                -1,
                lineType=cv2.LINE_AA,
            )
            # 키포인트 ID 텍스트 그리기
            cv2.putText(
                cv_image,
                str(kp.id),
                (int(kp.point.x), int(kp.point.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color_k,
                1,
                cv2.LINE_AA,
            )

        # 키포인트 ID로 위치를 찾는 헬퍼 함수
        def get_pk_pose(kp_id: int) -> Tuple[int]:
            for kp in keypoints_msg.data:
                if kp.id == kp_id:
                    return (int(kp.point.x), int(kp.point.y))
            return None

        # 스켈레톤 라인 그리기 (키포인트 간 연결)
        for i, sk in enumerate(ann.skeleton):
            kp1_pos = get_pk_pose(sk[0])
            kp2_pos = get_pk_pose(sk[1])

            if kp1_pos is not None and kp2_pos is not None:
                cv2.line(
                    cv_image,
                    kp1_pos,
                    kp2_pos,
                    [int(x) for x in ann.limb_color[i]],
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

        return cv_image

    # 3D 바운딩 박스 마커 생성 메서드
    # RViz에서 볼 수 있는 3D 바운딩 박스 마커 생성
    def create_bb_marker(self, detection: Detection, color: Tuple[int]) -> Marker:

        # 3D 바운딩 박스 정보 추출
        bbox3d = detection.bbox3d

        # 마커 메시지 초기화
        marker = Marker()
        marker.header.frame_id = bbox3d.frame_id

        # 마커 속성 설정
        marker.ns = "yolo_3d"
        marker.type = Marker.CUBE  # 큐브 타입 마커
        marker.action = Marker.ADD
        marker.frame_locked = False

        # 마커 위치 및 크기 설정
        marker.pose.position.x = bbox3d.center.position.x
        marker.pose.position.y = bbox3d.center.position.y
        marker.pose.position.z = bbox3d.center.position.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = bbox3d.size.x
        marker.scale.y = bbox3d.size.y
        marker.scale.z = bbox3d.size.z

        # 마커 색상 설정 (RGB 값을 0-1 범위로 정규화)
        marker.color.r = color[0] / 255.0
        marker.color.g = color[1] / 255.0
        marker.color.b = color[2] / 255.0
        marker.color.a = 0.4  # 투명도 설정

        # 마커 수명 설정 (0.5초)
        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = detection.class_name

        return marker

    # 3D 키포인트 마커 생성 메서드
    # RViz에서 볼 수 있는 3D 키포인트 마커 생성
    def create_kp_marker(self, keypoint: KeyPoint3D) -> Marker:

        # 마커 메시지 초기화
        marker = Marker()

        # 마커 속성 설정
        marker.ns = "yolo_3d"
        marker.type = Marker.SPHERE  # 구체 타입 마커
        marker.action = Marker.ADD
        marker.frame_locked = False

        # 마커 위치 설정
        marker.pose.position.x = keypoint.point.x
        marker.pose.position.y = keypoint.point.y
        marker.pose.position.z = keypoint.point.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05  # 구체 크기
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        # 마커 색상 설정 (신뢰도에 따라 색상 변경: 낮은 신뢰도=빨강, 높은 신뢰도=파랑)
        marker.color.r = (1.0 - keypoint.score) * 255.0
        marker.color.g = 0.0
        marker.color.b = keypoint.score * 255.0
        marker.color.a = 0.4  # 투명도 설정

        # 마커 수명 설정 (0.5초)
        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = str(keypoint.id)

        return marker

    # 감지 결과 콜백 메서드
    # 이미지와 감지 결과를 받아 시각화하고 발행
    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray) -> None:
        # 이미지 메시지를 OpenCV 이미지로 변환
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        # 마커 배열 초기화
        bb_marker_array = MarkerArray()
        kp_marker_array = MarkerArray()

        # 각 감지 결과 처리
        detection: Detection
        for detection in detection_msg.detections:

            # 클래스별 랜덤 색상 할당 (처음 보는 클래스인 경우)
            class_name = detection.class_name
            if class_name not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self._class_to_color[class_name] = (r, g, b)

            color = self._class_to_color[class_name]

            # 2D 시각화: 바운딩 박스, 마스크, 키포인트
            cv_image = self.draw_box(cv_image, detection, color)
            cv_image = self.draw_mask(cv_image, detection, color)
            cv_image = self.draw_keypoints(cv_image, detection)

            # 3D 바운딩 박스 마커 생성 (frame_id가 있는 경우)
            if detection.bbox3d.frame_id:
                marker = self.create_bb_marker(detection, color)
                marker.header.stamp = img_msg.header.stamp
                marker.id = len(bb_marker_array.markers)
                bb_marker_array.markers.append(marker)

            # 3D 키포인트 마커 생성 (frame_id가 있는 경우)
            if detection.keypoints3d.frame_id:
                for kp in detection.keypoints3d.data:
                    marker = self.create_kp_marker(kp)
                    marker.header.frame_id = detection.keypoints3d.frame_id
                    marker.header.stamp = img_msg.header.stamp
                    marker.id = len(kp_marker_array.markers)
                    kp_marker_array.markers.append(marker)

        # 시각화 결과 발행
        # - 디버그 이미지 (바운딩 박스, 마스크, 키포인트가 그려진 이미지)
        # - 바운딩 박스 마커 (RViz용)
        # - 키포인트 마커 (RViz용)
        self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, encoding="bgr8"))
        self._bb_markers_pub.publish(bb_marker_array)
        self._kp_markers_pub.publish(kp_marker_array)


# 메인 함수: 노드 초기화 및 실행
def main():
    rclpy.init()
    node = DebugNode()
    node.trigger_configure()  # 노드 구성
    node.trigger_activate()   # 노드 활성화
    rclpy.spin(node)          # 노드 실행
    node.destroy_node()       # 노드 정리
    rclpy.shutdown()          # ROS 종료
