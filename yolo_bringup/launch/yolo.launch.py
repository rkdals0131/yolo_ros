# ROS2 런치 파일로, 여러 매개변수(Launch Argument)를 선언하고, 조건에 따라 여러 노드를 실행하는 역할을 합니다.
# 이 파일은 Ultralytics YOLO 모델(및 관련 기능)을 ROS2 환경에서 실행하기 위한 설정을 제공합니다.
#
# Miguel Ángel González Santamarta의 저작권 및 GNU GPL 라이선스 하에 배포됩니다.

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    # run_yolo 함수: 실제로 각 노드를 생성 및 실행하는 부분.
    # 이 함수는 use_tracking과 use_3d (추적 및 3D 검출 활성화 여부)를 인자로 받습니다.
    def run_yolo(context: LaunchContext, use_tracking, use_3d):

        # 인자로 전달된 use_tracking, use_3d 값을 문자열로 평가(eval)하여 불리언으로 변환합니다.
        use_tracking = eval(context.perform_substitution(use_tracking))
        use_3d = eval(context.perform_substitution(use_3d))

        # 각 런치 인자들을 LaunchConfiguration으로 정의하고, DeclareLaunchArgument를 통해 기본값과 설명을 설정합니다.
        model_type = LaunchConfiguration("model_type")
        model_type_cmd = DeclareLaunchArgument(
            "model_type",
            default_value="YOLO",
            choices=["YOLO", "World"],
            description="Ultralytics 모델 타입 (YOLO, World 중 선택)",
        )

        model = LaunchConfiguration("model")
        model_cmd = DeclareLaunchArgument(
            "model",
            default_value="/home/user1/yolov12/pretrained_models/yolov11n.pt",
            description="모델 이름 또는 경로",
        )

        tracker = LaunchConfiguration("tracker")
        tracker_cmd = DeclareLaunchArgument(
            "tracker",
            default_value="bytetrack.yaml",
            description="추적기(tracker) 이름 또는 경로",
        )

        device = LaunchConfiguration("device")
        device_cmd = DeclareLaunchArgument(
            "device",
            default_value="cuda:0",
            description="사용할 디바이스 (GPU/CPU)",
        )

        enable = LaunchConfiguration("enable")
        enable_cmd = DeclareLaunchArgument(
            "enable",
            default_value="True",
            description="YOLO 기능을 활성화할지 여부",
        )

        threshold = LaunchConfiguration("threshold")
        threshold_cmd = DeclareLaunchArgument(
            "threshold",
            default_value="0.5",
            description="검출된 객체의 최소 확률(신뢰도) 임계값",
        )

        iou = LaunchConfiguration("iou")
        iou_cmd = DeclareLaunchArgument(
            "iou",
            default_value="0.7",
            description="Non-Maximum Suppression (NMS) 시 사용되는 IoU 임계값",
        )

        imgsz_height = LaunchConfiguration("imgsz_height")
        imgsz_height_cmd = DeclareLaunchArgument(
            "imgsz_height",
            default_value="480",
            description="추론 시 사용할 이미지 높이",
        )

        imgsz_width = LaunchConfiguration("imgsz_width")
        imgsz_width_cmd = DeclareLaunchArgument(
            "imgsz_width",
            default_value="640",
            description="추론 시 사용할 이미지 너비",
        )

        half = LaunchConfiguration("half")
        half_cmd = DeclareLaunchArgument(
            "half",
            default_value="False",
            description="FP16(half-precision) 추론 활성화 여부",
        )

        max_det = LaunchConfiguration("max_det")
        max_det_cmd = DeclareLaunchArgument(
            "max_det",
            default_value="300",
            description="이미지 당 허용되는 최대 검출 개수",
        )

        augment = LaunchConfiguration("augment")
        augment_cmd = DeclareLaunchArgument(
            "augment",
            default_value="False",
            description="테스트 시 증강(TTA) 활성화 여부",
        )

        agnostic_nms = LaunchConfiguration("agnostic_nms")
        agnostic_nms_cmd = DeclareLaunchArgument(
            "agnostic_nms",
            default_value="False",
            description="클래스에 구애받지 않는 NMS 활성화 여부",
        )

        retina_masks = LaunchConfiguration("retina_masks")
        retina_masks_cmd = DeclareLaunchArgument(
            "retina_masks",
            default_value="False",
            description="고해상도 세그멘테이션 마스크 사용 여부",
        )

        input_image_topic = LaunchConfiguration("input_image_topic")
        input_image_topic_cmd = DeclareLaunchArgument(
            "input_image_topic",
            default_value="/image_raw",
            description="입력 이미지 토픽 이름",
        )

        image_reliability = LaunchConfiguration("image_reliability")
        image_reliability_cmd = DeclareLaunchArgument(
            "image_reliability",
            default_value="1",
            choices=["0", "1", "2"],
            description="입력 이미지 토픽의 QoS 신뢰성 (0: 시스템 기본, 1: Reliable, 2: Best Effort)",
        )

        input_depth_topic = LaunchConfiguration("input_depth_topic")
        input_depth_topic_cmd = DeclareLaunchArgument(
            "input_depth_topic",
            default_value="/camera/depth/image_raw",
            description="입력 깊이 이미지 토픽 이름",
        )

        depth_image_reliability = LaunchConfiguration("depth_image_reliability")
        depth_image_reliability_cmd = DeclareLaunchArgument(
            "depth_image_reliability",
            default_value="1",
            choices=["0", "1", "2"],
            description="입력 깊이 이미지 토픽의 QoS 신뢰성",
        )

        input_depth_info_topic = LaunchConfiguration("input_depth_info_topic")
        input_depth_info_topic_cmd = DeclareLaunchArgument(
            "input_depth_info_topic",
            default_value="/camera/depth/camera_info",
            description="입력 깊이 카메라 정보 토픽 이름",
        )

        depth_info_reliability = LaunchConfiguration("depth_info_reliability")
        depth_info_reliability_cmd = DeclareLaunchArgument(
            "depth_info_reliability",
            default_value="1",
            choices=["0", "1", "2"],
            description="입력 깊이 카메라 정보 토픽의 QoS 신뢰성",
        )

        target_frame = LaunchConfiguration("target_frame")
        target_frame_cmd = DeclareLaunchArgument(
            "target_frame",
            default_value="base_link",
            description="3D 박스 변환에 사용할 타겟 프레임",
        )

        depth_image_units_divisor = LaunchConfiguration("depth_image_units_divisor")
        depth_image_units_divisor_cmd = DeclareLaunchArgument(
            "depth_image_units_divisor",
            default_value="1000",
            description="깊이 이미지 값을 미터로 변환할 때 사용할 나누는 값",
        )

        maximum_detection_threshold = LaunchConfiguration("maximum_detection_threshold")
        maximum_detection_threshold_cmd = DeclareLaunchArgument(
            "maximum_detection_threshold",
            default_value="0.3",
            description="z축에서 최대 검출 임계값",
        )

        namespace = LaunchConfiguration("namespace")
        namespace_cmd = DeclareLaunchArgument(
            "namespace",
            default_value="yolo",
            description="노드에 적용할 네임스페이스",
        )

        use_debug = LaunchConfiguration("use_debug")
        use_debug_cmd = DeclareLaunchArgument(
            "use_debug",
            default_value="True",
            description="디버그 노드 활성화 여부",
        )

        # 3D 검출 및 디버깅을 위한 토픽 설정 (추적 사용 여부에 따라 변경)
        detect_3d_detections_topic = "detections"
        debug_detections_topic = "detections"

        if use_tracking:
            detect_3d_detections_topic = "tracking"

        if use_tracking and not use_3d:
            debug_detections_topic = "tracking"
        elif use_3d:
            debug_detections_topic = "detections_3d"

        # YOLO 노드 생성: 모델, 매개변수, 토픽 remapping 등 설정
        yolo_node_cmd = Node(
            package="yolo_ros",
            executable="yolo_node",
            name="yolo_node",
            namespace=namespace,
            parameters=[
                {
                    "model_type": model_type,
                    "model": model,
                    "device": device,
                    "enable": enable,
                    "threshold": threshold,
                    "iou": iou,
                    "imgsz_height": imgsz_height,
                    "imgsz_width": imgsz_width,
                    "half": half,
                    "max_det": max_det,
                    "augment": augment,
                    "agnostic_nms": agnostic_nms,
                    "retina_masks": retina_masks,
                    "image_reliability": image_reliability,
                }
            ],
            remappings=[("image_raw", input_image_topic)],
        )

        # 추적 노드 생성 (use_tracking이 True인 경우에만 실행)
        tracking_node_cmd = Node(
            package="yolo_ros",
            executable="tracking_node",
            name="tracking_node",
            namespace=namespace,
            parameters=[{"tracker": tracker, "image_reliability": image_reliability}],
            remappings=[("image_raw", input_image_topic)],
            condition=IfCondition(PythonExpression([str(use_tracking)])),
        )

        # 3D 검출 노드 생성 (use_3d가 True인 경우에만 실행)
        detect_3d_node_cmd = Node(
            package="yolo_ros",
            executable="detect_3d_node",
            name="detect_3d_node",
            namespace=namespace,
            parameters=[
                {
                    "target_frame": target_frame,
                    "maximum_detection_threshold": maximum_detection_threshold,
                    "depth_image_units_divisor": depth_image_units_divisor,
                    "depth_image_reliability": depth_image_reliability,
                    "depth_info_reliability": depth_info_reliability,
                }
            ],
            remappings=[
                ("depth_image", input_depth_topic),
                ("depth_info", input_depth_info_topic),
                ("detections", detect_3d_detections_topic),
            ],
            condition=IfCondition(PythonExpression([str(use_3d)])),
        )

        # 디버그 노드 생성 (use_debug가 True인 경우에만 실행)
        debug_node_cmd = Node(
            package="yolo_ros",
            executable="debug_node",
            name="debug_node",
            namespace=namespace,
            parameters=[{"image_reliability": image_reliability}],
            remappings=[
                ("image_raw", input_image_topic),
                ("detections", debug_detections_topic),
            ],
            condition=IfCondition(PythonExpression([use_debug])),
        )

        # 생성한 노드들을 반환
        return (
            model_type_cmd,
            model_cmd,
            tracker_cmd,
            device_cmd,
            enable_cmd,
            threshold_cmd,
            iou_cmd,
            imgsz_height_cmd,
            imgsz_width_cmd,
            half_cmd,
            max_det_cmd,
            augment_cmd,
            agnostic_nms_cmd,
            retina_masks_cmd,
            input_image_topic_cmd,
            image_reliability_cmd,
            input_depth_topic_cmd,
            depth_image_reliability_cmd,
            input_depth_info_topic_cmd,
            depth_info_reliability_cmd,
            target_frame_cmd,
            depth_image_units_divisor_cmd,
            maximum_detection_threshold_cmd,
            namespace_cmd,
            use_debug_cmd,
            yolo_node_cmd,
            tracking_node_cmd,
            detect_3d_node_cmd,
            debug_node_cmd,
        )

    # use_tracking와 use_3d 런치 인자를 선언 (기본값 True)
    use_tracking = LaunchConfiguration("use_tracking")
    use_tracking_cmd = DeclareLaunchArgument(
        "use_tracking",
        default_value="False",
        description="추적 기능 활성화 여부",
    )

    use_3d = LaunchConfiguration("use_3d")
    use_3d_cmd = DeclareLaunchArgument(
        "use_3d",
        default_value="False",
        description="3D 검출 기능 활성화 여부",
    )

    # 최종 LaunchDescription에 런치 인자와 OpaqueFunction(노드 생성 함수)을 포함하여 반환합니다.
    return LaunchDescription(
        [
            use_tracking_cmd,
            use_3d_cmd,
            OpaqueFunction(function=run_yolo, args=[use_tracking, use_3d]),
        ]
    )
