from __future__ import annotations

"""
实时相机抽象接口。

目标：先接入 Mock Camera，后续可无缝替换 RealSenseCamera。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """单帧点云数据（统一为 xyz float32）。"""

    points_xyz: np.ndarray
    frame_id: int
    source_path: Optional[Path] = None


class IPointCloudCamera(ABC):
    """点云相机统一接口。"""

    @property
    @abstractmethod
    def source_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_running(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_frame(self) -> CameraFrame:
        raise NotImplementedError
