"""
分割器接口定义（里程碑 4）
设计统一语义分割抽象类，便于后续替换 MMDetection3D / 其他分割框架。

统一输出：
  - 每个点的类别 id（labels，shape=(N,)）
  - 类别 id -> 类别名 映射（id_to_name）
  - 类别 id -> 颜色 映射（id_to_color，RGB 归一化 0~1）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from app.utils.logger import get_logger

logger = get_logger("core.segmentor.base_segmentor")


SegmentInput = Union[np.ndarray, str, Path, dict]


@dataclass(frozen=True)
class SegmentationResult:
    """统一语义分割结果数据结构。"""

    labels: np.ndarray  # shape (N,), int32
    id_to_name: Dict[int, str]
    id_to_color: Dict[int, list]  # {id: [r,g,b]}，归一化 0~1


class BaseSegmentor(ABC):
    """
    统一分割器接口。

    输入支持：
      - numpy 点云数组：(N,K), K>=3
      - 点云文件路径：.bin / .pcd
      - 样本对象（dict）：至少包含 points 字段（numpy 数组）
    """

    def segment(self, input_data: SegmentInput) -> SegmentationResult:
        """
        统一入口：把输入转换为 (N,K) float32 点云数组，再调用 _segment_impl。
        """
        points = self._load_or_sanitize_points(input_data)
        return self._segment_impl(points)

    def _load_or_sanitize_points(self, input_data: SegmentInput) -> np.ndarray:
        """
        将输入规范化为 (N,K) float32 点云数组，K>=3。
        """
        if isinstance(input_data, dict):
            if "points" not in input_data:
                raise ValueError("样本对象缺少 points 字段")
            pts = np.asarray(input_data["points"])
        elif isinstance(input_data, np.ndarray):
            pts = np.asarray(input_data)
        else:
            # 文件路径：复用 IO 层，读取后得到 Open3D 点云（目前仅提供 xyz）
            from app.io.pointcloud_loader import load_pointcloud

            pcd = load_pointcloud(Path(input_data))
            pts = np.asarray(pcd.points)

        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"点云形状不合法，期望 (N,>=3)，实际为 {pts.shape}")

        return pts.astype(np.float32, copy=False)

    @abstractmethod
    def _segment_impl(self, points: np.ndarray) -> SegmentationResult:
        """
        子类实现：对 (N,K) 点云执行逐点语义分割，输出统一 SegmentationResult。
        """
        raise NotImplementedError

