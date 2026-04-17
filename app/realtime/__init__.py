"""实时模式模块（Mock Camera / RealSense Camera / 可扩展真实相机）"""

from .camera_interface import CameraFrame, CameraInfo, IPointCloudCamera
from .mock_camera import MockCamera
from .realtime_pipeline import RealtimePipeline, RealtimeResult

# RealSenseCamera 的模块文件始终存在；pyrealsense2 的实际导入推迟到 start() 内部，
# 因此这里直接暴露类引用不会因未安装 SDK 而影响离线模式加载。
from .realsense_camera import RealSenseCamera

__all__ = [
    "CameraFrame",
    "CameraInfo",
    "IPointCloudCamera",
    "MockCamera",
    "RealSenseCamera",
    "RealtimePipeline",
    "RealtimeResult",
]
