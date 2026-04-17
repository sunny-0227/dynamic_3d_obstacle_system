from __future__ import annotations

"""
实时流水线

职责：
  - 持有一个 IPointCloudCamera 实例和一个 FullPipeline 实例
  - 提供两种帧消费模式：
      read_raw()         — 仅读帧，不做分析；适合预览/帧率测试
      read_and_analyze() — 读帧 + 检测 + 分割 + 融合；适合实时分析
  - 对外屏蔽相机差异，调用方只需关心 RealtimeResult

扩展说明：
  - 替换相机只需在 AppController.start_realtime_mode() 中换一个
    IPointCloudCamera 子类传入，此文件无需改动
  - 若需要支持「只检测不分割」，可在此处新增 read_and_detect() 方法
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.fusion.result_fusion import FusedScene
from app.core.pipeline.full_pipeline import FullPipeline
from app.realtime.camera_interface import CameraFrame, CameraInfo, IPointCloudCamera
from app.utils.logger import get_logger

logger = get_logger("realtime.pipeline")


@dataclass(frozen=True)
class RealtimeResult:
    """
    单帧处理结果。

    字段说明：
      frame   — 原始帧数据（CameraFrame）
      scene   — 融合场景（FusedScene），仅在 read_and_analyze() 时非 None
      elapsed — 本帧处理耗时（秒），read_raw 时仅含读帧时间
    """

    frame: CameraFrame
    scene: Optional[FusedScene]
    elapsed: float = 0.0


class RealtimePipeline:
    """
    实时处理流水线：Camera → (可选)FullPipeline → RealtimeResult

    参数：
      camera        — 任意 IPointCloudCamera 实现（Mock / RealSense / ...）
      full_pipeline — FullPipeline 实例（检测 + 分割 + 融合）
    """

    def __init__(self, camera: IPointCloudCamera, full_pipeline: FullPipeline) -> None:
        self._camera = camera
        self._full_pipeline = full_pipeline

    # ------------------------------------------------------------------
    # 代理属性（方便外层查询相机状态，不暴露 _camera 引用）
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        """数据源名称，来自相机的 source_name 属性。"""
        return self._camera.source_name

    @property
    def is_running(self) -> bool:
        """当前相机是否处于运行状态。"""
        return self._camera.is_running

    @property
    def camera_info(self) -> CameraInfo:
        """返回相机元数据（帧数、文件索引、总文件数等）。"""
        return self._camera.camera_info

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        启动相机。
        异常会向上透传，由 AppController 捕获并提示用户。
        """
        logger.info("[RealtimePipeline] 启动相机：%s", self._camera.source_name)
        self._camera.start()
        logger.info("[RealtimePipeline] 相机已启动，共 %d 帧可用", self.camera_info.total_files)

    def stop(self) -> None:
        """停止相机（幂等）。"""
        logger.info("[RealtimePipeline] 停止相机：%s", self._camera.source_name)
        self._camera.stop()

    # ------------------------------------------------------------------
    # 帧消费模式
    # ------------------------------------------------------------------

    def read_raw(self) -> RealtimeResult:
        """
        仅读取下一帧，不执行任何算法分析。

        适用场景：
          - 预览模式（查看点云帧率/连通性）
          - 帧率基准测试

        返回：RealtimeResult（scene=None）
        异常：传递自 camera.get_next_frame()
        """
        t0 = time.perf_counter()

        frame = self._camera.get_next_frame()

        elapsed = time.perf_counter() - t0
        n_pts = int(frame.points_xyz.shape[0])

        logger.debug(
            "[RealtimePipeline] read_raw | 帧 #%d | 点数：%d | 耗时：%.1f ms",
            frame.frame_id,
            n_pts,
            elapsed * 1000,
        )

        return RealtimeResult(frame=frame, scene=None, elapsed=elapsed)

    def read_and_analyze(self) -> RealtimeResult:
        """
        读取下一帧并执行检测 + 分割 + 融合分析。

        适用场景：
          - 实时分析模式

        返回：RealtimeResult（scene 包含 FusedScene）
        异常：
          - camera.get_next_frame() 的异常向上透传
          - FullPipeline 内部异常向上透传（由 _RealtimeThread 捕获并停止）
        """
        t0 = time.perf_counter()

        frame = self._camera.get_next_frame()

        t_algo = time.perf_counter()
        # copy=True：避免 FullPipeline 内部操作意外修改 CameraFrame 里的原始数组
        pts = np.array(frame.points_xyz, dtype=np.float32)
        scene: Optional[FusedScene] = self._full_pipeline.run(pts)
        t_end = time.perf_counter()

        elapsed = t_end - t0
        algo_ms = (t_end - t_algo) * 1000
        read_ms = (t_algo - t0) * 1000
        n_pts = int(pts.shape[0])

        logger.debug(
            "[RealtimePipeline] read_and_analyze | 帧 #%d | 点数：%d | "
            "读帧：%.1f ms | 算法：%.1f ms | 总计：%.1f ms",
            frame.frame_id,
            n_pts,
            read_ms,
            algo_ms,
            elapsed * 1000,
        )

        return RealtimeResult(frame=frame, scene=scene, elapsed=elapsed)

    def __repr__(self) -> str:
        return (
            f"RealtimePipeline("
            f"camera={self._camera.source_name}, "
            f"running={self.is_running}"
            f")"
        )
