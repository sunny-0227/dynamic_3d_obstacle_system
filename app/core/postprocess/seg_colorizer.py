"""
分割结果着色器（里程碑 4）
将逐点 labels 与调色板 id_to_color 转为 Open3D 彩色点云，供可视化使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from app.utils.logger import get_logger

logger = get_logger("core.postprocess.seg_colorizer")


@dataclass(frozen=True)
class ColorizeOutput:
    points_xyz: np.ndarray  # (N,3)
    colors_rgb: np.ndarray  # (N,3) float 0~1


class SegColorizer:
    """将 labels 映射为每点颜色。"""

    def __init__(self, default_color: Optional[List[float]] = None):
        self._default_color = default_color or [0.6, 0.6, 0.6]

    def colorize(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        id_to_color: Dict[int, List[float]],
    ) -> ColorizeOutput:
        """
        输入：
          - points: (N,K) 或 (N,3)
          - labels: (N,)
          - id_to_color: {id: [r,g,b]}，0~1
        输出：
          - (N,3) xyz 与 (N,3) rgb
        """
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"points 形状不合法：{pts.shape}")
        xyz = pts[:, :3].astype(np.float32, copy=False)

        lab = np.asarray(labels).astype(np.int32, copy=False)
        if lab.ndim != 1 or lab.shape[0] != xyz.shape[0]:
            raise ValueError(f"labels 形状不匹配：{lab.shape} vs {xyz.shape}")

        colors = np.empty((xyz.shape[0], 3), dtype=np.float32)
        colors[:] = np.asarray(self._default_color, dtype=np.float32)

        # 避免逐点 Python 循环：按 label 分组写入颜色
        unique_labels = np.unique(lab) if lab.size > 0 else np.array([], dtype=np.int32)
        for lid in unique_labels.tolist():
            c = id_to_color.get(int(lid), self._default_color)
            mask = lab == lid
            colors[mask] = np.asarray(c, dtype=np.float32)

        colors = np.clip(colors, 0.0, 1.0)
        return ColorizeOutput(points_xyz=xyz, colors_rgb=colors)

    def to_open3d_pointcloud(self, output: ColorizeOutput):
        """将 colorize 输出转为 Open3D PointCloud（延迟 import，避免无 Open3D 环境报错）。"""
        try:
            import open3d as o3d
        except Exception as e:
            raise ImportError("缺少 open3d 依赖，无法生成 Open3D 点云") from e

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(output.points_xyz.astype(np.float64, copy=False))
        pcd.colors = o3d.utility.Vector3dVector(output.colors_rgb.astype(np.float64, copy=False))
        return pcd

