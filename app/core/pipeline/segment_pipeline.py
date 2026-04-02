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
    seg: SegmentationResult
    colored_pcd: Optional[object]  # Open3D PointCloud（若可用）


class SegmentPipeline:
    def __init__(self, segmentor: BaseSegmentor, colorizer: Optional[SegColorizer] = None):
        self._segmentor = segmentor
        self._colorizer = colorizer or SegColorizer()

    def run(self, input_data: SegmentInput) -> SegmentPipelineOutput:
        seg = self._segmentor.segment(input_data)

        # 尝试生成 Open3D 彩色点云（若 open3d 不可用则返回 None）
        colored_pcd = None
        try:
            points = self._segmentor._load_or_sanitize_points(input_data)  # 复用规范化逻辑
            out = self._colorizer.colorize(points, seg.labels, seg.id_to_color)
            colored_pcd = self._colorizer.to_open3d_pointcloud(out)
        except Exception as e:
            logger.warning("生成 Open3D 彩色点云失败（不影响分割 labels 输出）：%s", e)

        return SegmentPipelineOutput(seg=seg, colored_pcd=colored_pcd)

