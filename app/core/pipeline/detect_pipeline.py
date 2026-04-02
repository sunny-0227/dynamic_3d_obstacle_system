"""
检测 pipeline（里程碑 3）

职责：
  - 统一入口 run(points_or_path)，支持 numpy 点云数组 或点云文件路径
  - 调用 detector 执行检测
  - 调用 box_converter 做后处理与格式标准化
  - 返回统一的 List[DetectionBox]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

from app.core.detector.base_detector import BaseDetector, DetectionBox, PointsInput
from app.core.postprocess.box_converter import BoxConverter
from app.utils.logger import get_logger

logger = get_logger("core.pipeline.detect_pipeline")


class DetectPipeline:
    """
    检测流水线封装：Detector + Postprocess(转换器)。
    """

    def __init__(
        self,
        detector: BaseDetector,
        box_converter: BoxConverter | None = None,
    ) -> None:
        self.detector = detector
        self.box_converter = box_converter if box_converter is not None else BoxConverter()

    def run(self, points_or_path: PointsInput) -> List[DetectionBox]:
        """
        执行检测并返回统一检测结果。
        """
        logger.info("开始执行检测 pipeline")
        detections = self.detector.detect(points_or_path)
        detections_std = self.box_converter.convert(detections)
        logger.info("检测完成 | 检测框数: %d", len(detections_std))
        return detections_std

