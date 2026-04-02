"""
数据集适配包
将各公开数据集解析为统一帧描述结构，与算法层、IO 层解耦。
"""

from app.datasets.nuscenes_parser import NuScenesFrameRecord
from app.datasets.nuscenes_loader import NuScenesMiniLoader

__all__ = ["NuScenesFrameRecord", "NuScenesMiniLoader"]
