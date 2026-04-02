"""
检测器接口定义（里程碑 3）
设计统一检测器抽象类，使后续替换 OpenPCDet / MMDetection3D 时保持算法层契约一致。

统一输出格式：
  - 类别名
  - 置信度
  - 中心点 center (x, y, z)
  - 尺寸 size (长, 宽, 高) 对应 l, w, h
  - yaw（绕 Z 轴旋转角，弧度）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np


PointsInput = Union[np.ndarray, str, Path]


@dataclass(frozen=True)
class DetectionBox:
    """标准化 3D 检测框数据结构。"""

    class_name: str
    score: float
    center: np.ndarray  # (3,)
    size: np.ndarray  # (3,) [l, w, h]
    yaw: float


class BaseDetector(ABC):
    """
    统一检测器接口。

    - 输入：支持 numpy 点云数组 或点云文件路径（.bin/.pcd）
    - 输出：List[DetectionBox]
    """

    def detect(self, points_or_path: PointsInput) -> List[DetectionBox]:
        """
        统一入口：将输入点云转换为 numpy 数组后调用 _detect_impl。

        重要说明：
            为了兼容真实 3D 检测模型（如 OpenPCDet），此处不再强制裁剪为 (N,3)；
            而是保留原始点特征维度 (N,K)，其中 K>=3（xyz + 可选强度/环号等）。
            子类 detector 可根据自身需求决定是否只取 xyz 或使用额外特征。
        """
        points = self._load_or_sanitize_points(points_or_path)
        return self._detect_impl(points)

    def _load_or_sanitize_points(self, points_or_path: PointsInput) -> np.ndarray:
        """
        将输入规范化为 (N, K) float32 点云数组，其中 K>=3。
        """
        if isinstance(points_or_path, np.ndarray):
            pts = np.asarray(points_or_path)
        else:
            # 文件路径：交给 IO 层读取（仅用于将点云变为算法需要的数据）
            from app.io.pointcloud_loader import load_pointcloud

            path = Path(points_or_path)
            pcd = load_pointcloud(path)
            pts = np.asarray(pcd.points)

        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(
                f"点云形状不合法，期望 (N, >=3)；实际为 {pts.shape}"
            )

        # 保留所有列（至少 xyz），避免丢失强度等关键特征
        pts_all = pts.astype(np.float32, copy=False)
        return pts_all

    @abstractmethod
    def _detect_impl(self, points: np.ndarray) -> List[DetectionBox]:
        """
        子类只需实现：对 (N,K) 点云执行检测并返回标准化检测框列表。

        参数：
            points: shape (N,K)，K>=3（xyz + 可选强度等）
        """
        raise NotImplementedError

