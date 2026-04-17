from __future__ import annotations

"""
实时点云相机抽象接口

设计目标：
  - 先接入 Mock Camera，后续可无缝替换为 RealSenseCamera 或其他真实相机
  - 所有相机实现必须继承 IPointCloudCamera 并实现全部抽象方法
  - CameraFrame 作为帧数据的统一载体，跨相机类型保持一致

接口方法：
  start()           — 初始化并启动相机（分配资源、开始流）
  stop()            — 停止相机并释放资源
  get_next_frame()  — 阻塞或非阻塞获取下一帧；尚未启动时应抛出 RuntimeError
  read_frame()      — get_next_frame 的别名，保持向后兼容

扩展指引：
  新增 RealSenseCamera 只需：
    class RealSenseCamera(IPointCloudCamera):
        def start(self): ...
        def stop(self): ...
        def get_next_frame(self) -> CameraFrame: ...
  无需修改 RealtimePipeline 或 controller。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """
    单帧点云数据（统一表示，与相机类型无关）。

    字段说明：
      points_xyz  — 形状 (N, 3) float32 的点坐标数组
      frame_id    — 从 1 开始的单调递增帧序号（每次 start() 后重置）
      total_files — 该数据源的总文件数（Mock 时为目录文件数；真实相机填 0 表示未知）
      source_path — 原始文件路径（Mock 时有效；真实相机可为 None）
      timestamp   — 采集时间戳（秒，float）；若相机无法提供则为 None
    """

    points_xyz: np.ndarray
    frame_id: int
    total_files: int = 0
    source_path: Optional[Path] = None
    timestamp: Optional[float] = None


@dataclass
class CameraInfo:
    """
    相机元数据（只读属性的结构化表示）。

    字段说明：
      source_name   — 数据源名称，如 "Mock" / "RealSense D435i"
      is_running    — 当前是否处于采集状态
      frame_count   — 已输出的帧数（自 start() 起累计）
      current_index — Mock 场景下当前文件索引（0-based）；真实相机填 -1
      total_files   — Mock 场景下目录文件数；真实相机填 0
    """

    source_name: str = "Unknown"
    is_running: bool = False
    frame_count: int = 0
    current_index: int = -1
    total_files: int = 0


class IPointCloudCamera(ABC):
    """
    点云相机统一抽象接口。

    子类需实现：start / stop / get_next_frame
    子类可选覆盖：source_name / is_running / camera_info
    """

    # ------------------------------------------------------------------
    # 只读属性（子类可覆盖为 @property）
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def source_name(self) -> str:
        """相机/数据源名称，如 "Mock" / "RealSense D435i"。"""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """当前是否处于运行（采集）状态。"""

    @property
    def camera_info(self) -> CameraInfo:
        """返回相机当前元数据；子类可覆盖以提供更多信息。"""
        return CameraInfo(
            source_name=self.source_name,
            is_running=self.is_running,
        )

    # ------------------------------------------------------------------
    # 核心生命周期方法
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self) -> None:
        """
        初始化并启动相机。
        应在第一次调用 get_next_frame() 之前调用。
        若相机已启动，可选择幂等处理或抛出 RuntimeError。
        """

    @abstractmethod
    def stop(self) -> None:
        """
        停止相机并释放资源。
        应确保可安全重复调用（幂等）。
        """

    @abstractmethod
    def get_next_frame(self) -> CameraFrame:
        """
        获取下一帧点云数据。

        返回：CameraFrame（含 points_xyz / frame_id / source_path 等信息）
        异常：
          RuntimeError  — 相机尚未启动
          StopIteration — 数据源已耗尽且不循环（Mock 循环模式不会触发）
          IOError       — 读取过程中发生 I/O 错误
        """

    def read_frame(self) -> CameraFrame:
        """
        get_next_frame 的别名，保持对旧版 pipeline 的向后兼容。
        不建议在新代码中直接使用，请统一调用 get_next_frame()。
        """
        return self.get_next_frame()
