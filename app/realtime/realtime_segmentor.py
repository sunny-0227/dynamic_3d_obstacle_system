from __future__ import annotations

"""实时分割适配层：复用现有 SegmentPipeline。"""

import numpy as np

from app.core.pipeline.segment_pipeline import SegmentPipeline, SegmentPipelineOutput


class RealtimeSegmentor:
    def __init__(self, segment_pipeline: SegmentPipeline):
        self._segment_pipeline = segment_pipeline

    def run(self, points_xyz: np.ndarray) -> SegmentPipelineOutput:
        return self._segment_pipeline.run(points_xyz)
