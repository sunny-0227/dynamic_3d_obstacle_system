"""
分割 Pipeline（里程碑 4）
串联：输入点云 -> 分割器 -> 着色（可选）-> 统一输出
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.postprocess.seg_colorizer import SegColorizer
from app.core.segmentor.base_segmentor import BaseSegmentor, SegmentationResult, SegmentInput
from app.utils.logger import get_logger

logger = get_logger("core.pipeline.segment_pipeline")


@dataclass(frozen=True)
class SegmentPipelineOutput:
    # 分割时实际使用的点（用于保证 labels 与点集一一对应，避免后续真实模型下采样导致错位）
    points_xyz: np.ndarray  # (N,3) float32
    seg: SegmentationResult
    colored_pcd: Optional[object]  # Open3D PointCloud（若可用）


class SegmentPipeline:
    def __init__(self, segmentor: BaseSegmentor, colorizer: Optional[SegColorizer] = None):
        self._segmentor = segmentor
        self._colorizer = colorizer or SegColorizer()

    def run(self, input_data: SegmentInput) -> SegmentPipelineOutput:
        # 统一取一次“实际点集”，后续所有输出都围绕该点集，避免错位
        points = self._segmentor._load_or_sanitize_points(input_data)  # (N,K)
        points_xyz = np.asarray(points[:, :3], dtype=np.float32)
        seg = self._segmentor._segment_impl(points)

        # 尝试生成 Open3D 彩色点云（若 open3d 不可用则返回 None）
        colored_pcd = None
        try:
            out = self._colorizer.colorize(points, seg.labels, seg.id_to_color)
            colored_pcd = self._colorizer.to_open3d_pointcloud(out)
        except Exception as e:
            logger.warning("生成 Open3D 彩色点云失败（不影响分割 labels 输出）：%s", e)

        return SegmentPipelineOutput(points_xyz=points_xyz, seg=seg, colored_pcd=colored_pcd)

