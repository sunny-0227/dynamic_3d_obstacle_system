from __future__ import annotations

"""实时检测适配层：复用现有 DetectPipeline。"""

from typing import List

import numpy as np

from app.core.detector.base_detector import DetectionBox
from app.core.pipeline.detect_pipeline import DetectPipeline


class RealtimeDetector:
    def __init__(self, detect_pipeline: DetectPipeline):
        self._detect_pipeline = detect_pipeline

    def run(self, points_xyz: np.ndarray) -> List[DetectionBox]:
        return self._detect_pipeline.run(points_xyz)
