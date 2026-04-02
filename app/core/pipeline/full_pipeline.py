"""
完整融合 Pipeline（里程碑 5）

目标：
  - 一键运行：检测 + 分割 + 结果坐标融合 + 可视化输入准备
  - 数据流清晰：输入点云 -> 各模块 -> FusedScene
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.detector.base_detector import DetectionBox, PointsInput
from app.core.fusion.result_fusion import CoordinateSystemSpec, FusedScene, ResultFusion
from app.core.geometry.transform import Transform
from app.core.pipeline.detect_pipeline import DetectPipeline
from app.core.pipeline.segment_pipeline import SegmentPipeline, SegmentPipelineOutput
from app.utils.logger import get_logger

logger = get_logger("core.pipeline.full_pipeline")


@dataclass(frozen=True)
class FullPipelineConfig:
    """
    预留：用于未来把坐标系变换（例如 sensor->ego->global）注入到 pipeline。
    当前占位默认所有输出与输入点云坐标一致。
    """

    coord_spec: CoordinateSystemSpec = CoordinateSystemSpec()
    T_det_to_points: Optional[Transform] = None
    T_seg_to_points: Optional[Transform] = None


class FullPipeline:
    def __init__(
        self,
        detect_pipeline: DetectPipeline,
        segment_pipeline: SegmentPipeline,
        fusion: Optional[ResultFusion] = None,
        cfg: Optional[FullPipelineConfig] = None,
    ):
        self._detect = detect_pipeline
        self._segment = segment_pipeline
        self._fusion = fusion or ResultFusion()
        self._cfg = cfg or FullPipelineConfig()

    def run(self, points_or_path: PointsInput) -> FusedScene:
        """
        输入：
          - numpy 点云数组 (N,3) 或 (N,K)
          - 或点云文件路径（.bin/.pcd）

        输出：
          - FusedScene（统一坐标，可直接用于 Open3D 同窗渲染）
        """
        logger.info("开始执行完整 pipeline：分割 + 检测 + 融合")

        seg_out: SegmentPipelineOutput = self._segment.run(points_or_path)

        # 检测与分割都基于同一输入（points_or_path）执行
        det_boxes: list[DetectionBox] = self._detect.run(points_or_path)

        # 使用分割 pipeline 实际使用的点集，保证 labels 与点集一一对应（避免未来真实模型下采样错位）
        points_xyz = seg_out.points_xyz

        fused = self._fusion.fuse(
            points_xyz=points_xyz,
            seg_out=seg_out,
            detections=det_boxes,
            T_det_to_points=self._cfg.T_det_to_points,
            T_seg_to_points=self._cfg.T_seg_to_points,
            coord_spec=self._cfg.coord_spec,
        )
        logger.info(
            "完整 pipeline 完成 | 点数=%d | 检测框=%d",
            fused.points_xyz.shape[0],
            len(fused.detections),
        )
        return fused

