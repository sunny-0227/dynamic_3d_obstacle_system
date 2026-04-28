"""检测 pipeline：调用 BaseDetector 并通过 BoxConverter 标准化输出。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

import time

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
        """执行检测并返回统一检测结果。"""
        det_cls = type(self.detector).__name__
        logger.info("[DetectPipeline] 开始检测 | 检测器=%s", det_cls)
        t0 = time.perf_counter()
        detections = self.detector.detect(points_or_path)
        detections_std = self.box_converter.convert(detections)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "[DetectPipeline] 检测完成 | 检测器=%s | 框数=%d | 耗时=%.0f ms",
            det_cls, len(detections_std), elapsed,
        )
        return detections_std

