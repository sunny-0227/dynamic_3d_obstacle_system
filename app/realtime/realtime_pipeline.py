from __future__ import annotations

"""
实时流水线：从相机读取 ->（可选）检测/分割/融合 -> 输出实时结果。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.fusion.result_fusion import FusedScene
from app.core.pipeline.full_pipeline import FullPipeline
from app.realtime.camera_interface import CameraFrame, IPointCloudCamera


@dataclass(frozen=True)
class RealtimeResult:
    frame: CameraFrame
    scene: Optional[FusedScene]


class RealtimePipeline:
    def __init__(self, camera: IPointCloudCamera, full_pipeline: FullPipeline):
        self._camera = camera
        self._full_pipeline = full_pipeline

    @property
    def source_name(self) -> str:
        return self._camera.source_name

    @property
    def is_running(self) -> bool:
        return self._camera.is_running

    def start(self) -> None:
        self._camera.start()

    def stop(self) -> None:
        self._camera.stop()

    def read_raw(self) -> RealtimeResult:
        frame = self._camera.read_frame()
        return RealtimeResult(frame=frame, scene=None)

    def read_and_analyze(self) -> RealtimeResult:
        frame = self._camera.read_frame()
        scene = self._full_pipeline.run(np.asarray(frame.points_xyz, dtype=np.float32))
        return RealtimeResult(frame=frame, scene=scene)
