"""
融合模块包（里程碑 1 + 里程碑 5）

说明：
  - 里程碑 1 的 `run_full_pipeline` / `FusionResult` 与 `fake_detector` / `fake_segmentor` 直接对接，
    供独立演示或脚本使用；当前 PyQt 主流程走 `DetectPipeline` / `SegmentPipeline` / `ResultFusion.fuse`，
    占位推理在 `OpenPCDetDetector._detect_fake` 与 `MMDet3DSegmentor._segment_fake_impl` 中实现。
  - 里程碑1 的“伪分割 + 伪检测”融合接口历史上位于 `app/core/fusion.py`。
    为满足里程碑5 的包结构 `app/core/fusion/result_fusion.py`，这里将其迁移到包内并保持 API 不变：
      - FusionResult
      - run_full_pipeline
  - 里程碑5 的坐标融合入口位于：
      - app.core.fusion.result_fusion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import open3d as o3d

from app.core.fake_detector import DetectionResult, run_fake_detection
from app.core.fake_segmentor import run_fake_segmentation
from app.utils.logger import get_logger

logger = get_logger("core.fusion")


@dataclass
class FusionResult:
    """
    里程碑1：融合结果数据类，包含原始点云、分割着色点云和3D检测框列表。
    """

    raw_pcd: o3d.geometry.PointCloud
    segmented_pcd: o3d.geometry.PointCloud
    seg_labels: np.ndarray
    detections: List[DetectionResult] = field(default_factory=list)

    def get_all_geometries(self) -> List[o3d.geometry.Geometry3D]:
        geometries: List[o3d.geometry.Geometry3D] = [self.segmented_pcd]
        for det in self.detections:
            if det.box_geometry is not None:
                geometries.append(det.box_geometry)
        return geometries


def run_full_pipeline(
    pcd: o3d.geometry.PointCloud,
    num_boxes: int = 3,
    score_range: tuple = (0.75, 0.99),
    num_classes: int = 4,
    class_colors: dict = None,
) -> FusionResult:
    """
    里程碑1：执行完整的伪推理流水线：语义分割 + 3D目标检测，并将结果融合。
    """
    logger.info("开始完整推理流水线")

    logger.info("步骤1/2：语义分割")
    segmented_pcd, seg_labels = run_fake_segmentation(
        pcd, num_classes=num_classes, class_colors=class_colors
    )

    logger.info("步骤2/2：3D目标检测")
    detections = run_fake_detection(pcd, num_boxes=num_boxes, score_range=score_range)

    result = FusionResult(
        raw_pcd=pcd,
        segmented_pcd=segmented_pcd,
        seg_labels=seg_labels,
        detections=detections,
    )
    logger.info("流水线完成 | 分割点数: %d | 检测框数: %d", len(seg_labels), len(detections))
    return result


__all__ = [
    "FusionResult",
    "run_full_pipeline",
]

