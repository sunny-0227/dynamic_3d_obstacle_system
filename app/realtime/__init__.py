"""实时模式模块（Mock Camera / RealSense Camera / 可扩展真实相机）"""

from .camera_interface import CameraFrame, CameraInfo, IPointCloudCamera
from .mock_camera import MockCamera
from .realtime_detector import (
    ClusterConfig,
    ClusterResult,
    LightweightDetector,
    RealtimeDetector,
)
from .realtime_pipeline import (
    LightweightRealtimePipeline,
    RealtimePipeline,
    RealtimeResult,
)
from .realtime_segmentor import (
    GroundSegConfig,
    LightweightSegmentor,
    RealtimeSegmentor,
    SEG_BACKGROUND,
    SEG_GROUND,
    SEG_NOISE,
    SEG_OBSTACLE,
)

# RealSenseCamera 的模块文件始终存在；pyrealsense2 的实际导入推迟到 start() 内部，
# 因此这里直接暴露类引用不会因未安装 SDK 而影响离线模式加载。
from .realsense_camera import RealSenseCamera

__all__ = [
    # 相机接口
    "CameraFrame",
    "CameraInfo",
    "IPointCloudCamera",
    # 相机实现
    "MockCamera",
    "RealSenseCamera",
    # 流水线
    "RealtimePipeline",
    "LightweightRealtimePipeline",
    "RealtimeResult",
    # 轻量分割
    "LightweightSegmentor",
    "RealtimeSegmentor",
    "GroundSegConfig",
    "SEG_BACKGROUND",
    "SEG_GROUND",
    "SEG_OBSTACLE",
    "SEG_NOISE",
    # 轻量检测
    "LightweightDetector",
    "RealtimeDetector",
    "ClusterConfig",
    "ClusterResult",
]
