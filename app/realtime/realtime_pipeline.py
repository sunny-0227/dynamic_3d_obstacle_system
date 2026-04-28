"""
实时点云处理流水线（双模式）

性能关键点：
  - voxel_downsample + random_subsample 控制算法输入点数
  - process_interval：每 N 帧执行一次算法，提升显示帧率
  - RealtimeResult 分别携带相机 FPS 与算法处理 FPS
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.core.detector.base_detector import DetectionBox
from app.core.fusion.result_fusion import (
    CoordinateSystemSpec,
    FusedScene,
    ResultFusion,
)
from app.core.pipeline.full_pipeline import FullPipeline
from app.core.pipeline.segment_pipeline import SegmentPipelineOutput
from app.realtime.camera_interface import CameraFrame, CameraInfo, IPointCloudCamera
from app.realtime.realtime_detector import (
    ClusterResult,
    LightweightDetector,
    ClusterConfig,
)
from app.realtime.realtime_segmentor import (
    LightweightSegmentor,
    GroundSegConfig,
    SEG_OBSTACLE,
)
from app.utils.logger import get_logger

logger = get_logger("realtime.pipeline")


# ---------------------------------------------------------------------------
# 共用结果数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RealtimeResult:
    """单帧处理结果（两种 pipeline 共用）。"""
    frame: CameraFrame
    scene: Optional[FusedScene]
    clusters: List[ClusterResult] = field(default_factory=list)
    elapsed: float = 0.0
    camera_fps: float = 0.0    # 相机采集帧率（Hz）
    process_fps: float = 0.0   # 算法处理帧率（Hz），非分析帧为 0
    raw_points: int = 0        # 原始点数（下采样前）
    proc_points: int = 0       # 算法处理点数（下采样后）
    proc_elapsed: float = 0.0  # 算法耗时（秒），不含读帧时间


# ---------------------------------------------------------------------------
# 点云下采样工具函数
# ---------------------------------------------------------------------------

def _voxel_downsample(pts_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """体素下采样，优先使用 Open3D，降级为手动网格法。0 或负数表示不下采样。"""
    if voxel_size <= 0 or pts_xyz.shape[0] == 0:
        return pts_xyz
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz.astype(np.float64))
        return np.asarray(pcd.voxel_down_sample(voxel_size).points, dtype=np.float32)
    except Exception:
        pass
    vox_idx = np.floor(pts_xyz / voxel_size).astype(np.int32)
    seen: dict = {}
    keep = []
    for i, vi in enumerate(map(tuple, vox_idx)):
        if vi not in seen:
            seen[vi] = i
            keep.append(i)
    return pts_xyz[keep]


def _random_subsample(pts_xyz: np.ndarray, max_points: int) -> np.ndarray:
    """随机抽稀：当点数超过 max_points 时随机选取 max_points 个点。"""
    if max_points <= 0 or pts_xyz.shape[0] <= max_points:
        return pts_xyz
    idx = np.random.choice(pts_xyz.shape[0], size=max_points, replace=False)
    return pts_xyz[idx]


class RealtimePipeline:
    """
    实时处理流水线：Camera → (可选)FullPipeline → RealtimeResult

    参数：
      camera              — 任意 IPointCloudCamera 实现
      full_pipeline       — FullPipeline 实例
      voxel_size          — 点云下采样体素大小（米），0 表示不下采样
      max_points_for_proc — 下采样后仍超量则随机再抽稀
      max_points_for_disp — 送显示的最大点数
      process_interval    — 每 N 帧分析一次（1=每帧均分析）
    """

    def __init__(
        self,
        camera: IPointCloudCamera,
        full_pipeline: FullPipeline,
        voxel_size: float = 0.0,
        max_points_for_proc: int = 0,
        max_points_for_disp: int = 0,
        process_interval: int = 1,
    ) -> None:
        self._camera = camera
        self._full_pipeline = full_pipeline
        self._voxel_size = float(voxel_size)
        self._max_proc = int(max_points_for_proc)
        self._max_disp = int(max_points_for_disp)
        self._proc_interval = max(1, int(process_interval))
        self._frame_counter = 0        # 内部帧计数（用于 process_interval 判断）
        self._prev_cam_t: float = 0.0  # 上一帧时间（计算相机 FPS）
        self._prev_proc_t: float = 0.0 # 上一次分析时间（计算处理 FPS）

    @property
    def source_name(self) -> str:
        return self._camera.source_name

    @property
    def is_running(self) -> bool:
        return self._camera.is_running

    @property
    def camera_info(self) -> CameraInfo:
        return self._camera.camera_info

    def start(self) -> None:
        logger.info("[RealtimePipeline] 启动相机：%s", self._camera.source_name)
        self._camera.start()
        self._frame_counter = 0
        self._prev_cam_t = time.perf_counter()

    def stop(self) -> None:
        logger.info("[RealtimePipeline] 停止相机：%s", self._camera.source_name)
        self._camera.stop()

    def read_raw(self) -> RealtimeResult:
        """仅读取下一帧，不执行算法分析。"""
        t0 = time.perf_counter()
        frame = self._camera.get_next_frame()
        now = time.perf_counter()

        cam_fps = 1.0 / max(now - self._prev_cam_t, 1e-6)
        self._prev_cam_t = now
        raw_pts = int(frame.points_xyz.shape[0])

        elapsed = now - t0
        return RealtimeResult(
            frame=frame, scene=None, elapsed=elapsed,
            camera_fps=cam_fps, process_fps=0.0,
            raw_points=raw_pts, proc_points=raw_pts,
        )

    def read_and_analyze(self) -> RealtimeResult:
        """读取帧并（按 process_interval）执行分析。"""
        t0 = time.perf_counter()
        frame = self._camera.get_next_frame()
        now = time.perf_counter()

        cam_fps = 1.0 / max(now - self._prev_cam_t, 1e-6)
        self._prev_cam_t = now
        self._frame_counter += 1
        raw_pts = int(frame.points_xyz.shape[0])

        # 判断本帧是否执行算法
        do_process = (self._frame_counter % self._proc_interval == 0)

        if not do_process:
            return RealtimeResult(
                frame=frame, scene=None, elapsed=now - t0,
                camera_fps=cam_fps, process_fps=0.0,
                raw_points=raw_pts, proc_points=0,
            )

        # 下采样
        pts = np.array(frame.points_xyz, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        pts = pts[:, :3]
        pts = _voxel_downsample(pts, self._voxel_size)
        pts = _random_subsample(pts, self._max_proc)
        proc_pts = int(pts.shape[0])

        t_algo = time.perf_counter()
        scene: Optional[FusedScene] = self._full_pipeline.run(pts)
        t_end = time.perf_counter()

        proc_fps = 1.0 / max(t_end - self._prev_proc_t, 1e-6)
        self._prev_proc_t = t_end
        proc_elapsed = t_end - t_algo

        logger.debug(
            "[RealtimePipeline] 帧 #%d | 原始=%d 下采=%d | 相机%.1ffps 处理%.1ffps | 算法%.0fms",
            frame.frame_id, raw_pts, proc_pts, cam_fps, proc_fps, proc_elapsed * 1000,
        )

        return RealtimeResult(
            frame=frame, scene=scene,
            elapsed=t_end - t0,
            camera_fps=cam_fps, process_fps=proc_fps,
            raw_points=raw_pts, proc_points=proc_pts,
            proc_elapsed=proc_elapsed,
        )

    def __repr__(self) -> str:
        return (
            f"RealtimePipeline(camera={self._camera.source_name}, "
            f"voxel={self._voxel_size}, interval={self._proc_interval})"
        )


class LightweightRealtimePipeline:
    """
    轻量级实时处理流水线：Camera → RANSAC 地面分割 → DBSCAN 聚类检测 → FusedScene

    不依赖 GPU 和深度学习模型，适合实时演示。
    通过 voxel_size / max_points_for_proc / process_interval 控制处理负载。
    """

    def __init__(
        self,
        camera: IPointCloudCamera,
        seg_config: Optional[GroundSegConfig] = None,
        det_config: Optional[ClusterConfig] = None,
        voxel_size: float = 0.05,
        max_points_for_proc: int = 20000,
        process_interval: int = 1,
    ) -> None:
        self._camera = camera
        self._segmentor = LightweightSegmentor(config=seg_config)
        self._detector = LightweightDetector(config=det_config)
        self._fusion = ResultFusion()
        self._voxel_size = float(voxel_size)
        self._max_proc = int(max_points_for_proc)
        self._proc_interval = max(1, int(process_interval))
        self._frame_counter = 0
        self._prev_cam_t: float = 0.0
        self._prev_proc_t: float = 0.0

    @property
    def source_name(self) -> str:
        return self._camera.source_name

    @property
    def is_running(self) -> bool:
        return self._camera.is_running

    @property
    def camera_info(self) -> CameraInfo:
        return self._camera.camera_info

    def start(self) -> None:
        logger.info("[LightweightRealtimePipeline] 启动相机：%s", self._camera.source_name)
        self._camera.start()
        self._frame_counter = 0
        self._prev_cam_t = time.perf_counter()

    def stop(self) -> None:
        logger.info("[LightweightRealtimePipeline] 停止相机：%s", self._camera.source_name)
        self._camera.stop()

    def read_raw(self) -> RealtimeResult:
        """仅读取下一帧，不执行分割/检测。"""
        t0 = time.perf_counter()
        frame = self._camera.get_next_frame()
        now = time.perf_counter()
        cam_fps = 1.0 / max(now - self._prev_cam_t, 1e-6)
        self._prev_cam_t = now
        raw_pts = int(frame.points_xyz.shape[0])
        return RealtimeResult(
            frame=frame, scene=None, clusters=[],
            elapsed=now - t0,
            camera_fps=cam_fps, process_fps=0.0,
            raw_points=raw_pts, proc_points=raw_pts,
        )

    def read_and_analyze(self) -> RealtimeResult:
        """读取一帧并（按 process_interval）执行轻量分割 + 聚类检测 + 融合。"""
        t0 = time.perf_counter()

        frame = self._camera.get_next_frame()
        now = time.perf_counter()
        cam_fps = 1.0 / max(now - self._prev_cam_t, 1e-6)
        self._prev_cam_t = now
        self._frame_counter += 1

        raw_pts_xyz = np.array(frame.points_xyz, dtype=np.float32)
        if raw_pts_xyz.ndim == 1:
            raw_pts_xyz = raw_pts_xyz.reshape(-1, 3)
        raw_pts_xyz = raw_pts_xyz[:, :3]
        raw_n = int(raw_pts_xyz.shape[0])

        do_process = (self._frame_counter % self._proc_interval == 0)
        if not do_process:
            return RealtimeResult(
                frame=frame, scene=None, clusters=[],
                elapsed=time.perf_counter() - t0,
                camera_fps=cam_fps, process_fps=0.0,
                raw_points=raw_n, proc_points=0,
            )

        pts_xyz = _voxel_downsample(raw_pts_xyz, self._voxel_size)
        pts_xyz = _random_subsample(pts_xyz, self._max_proc)
        proc_n = int(pts_xyz.shape[0])

        t_algo = time.perf_counter()

        seg_result, colored_pcd = self._segmentor.segment_with_colored_pcd(pts_xyz)
        seg_out = SegmentPipelineOutput(
            points_xyz=pts_xyz,
            seg=seg_result,
            colored_pcd=colored_pcd,
        )

        t2 = time.perf_counter()

        obstacle_mask = seg_result.labels == SEG_OBSTACLE
        obstacle_pts = pts_xyz[obstacle_mask]

        detections: List[DetectionBox] = []
        clusters: List[ClusterResult] = []
        det_obbs: Optional[list] = None

        if obstacle_pts.shape[0] >= self._detector._cfg.cluster_min_points:
            detections, clusters, det_obbs = self._detector.detect_with_obbs(obstacle_pts)
        else:
            logger.debug(
                "[LightweightRealtimePipeline] 障碍物候选点不足（%d），跳过聚类",
                obstacle_pts.shape[0],
            )

        t3 = time.perf_counter()

        scene = self._build_fused_scene(
            pts_xyz=pts_xyz,
            seg_out=seg_out,
            detections=detections,
            det_obbs=det_obbs,
        )

        t_end = time.perf_counter()
        proc_fps = 1.0 / max(t_end - self._prev_proc_t, 1e-6)
        self._prev_proc_t = t_end
        proc_elapsed = t_end - t_algo

        logger.debug(
            "[LightweightRealtimePipeline] 帧 #%d | 原始=%d 下采=%d | "
            "相机%.1ffps 处理%.1ffps | 分割%.0fms 检测%.0fms 总%.0fms",
            frame.frame_id, raw_n, proc_n,
            cam_fps, proc_fps,
            (t2 - t_algo) * 1000, (t3 - t2) * 1000, proc_elapsed * 1000,
        )

        return RealtimeResult(
            frame=frame,
            scene=scene,
            clusters=clusters,
            elapsed=t_end - t0,
            camera_fps=cam_fps,
            process_fps=proc_fps,
            raw_points=raw_n,
            proc_points=proc_n,
            proc_elapsed=proc_elapsed,
        )

    def _build_fused_scene(
        self,
        pts_xyz: np.ndarray,
        seg_out: SegmentPipelineOutput,
        detections: List[DetectionBox],
        det_obbs: Optional[list],
    ) -> FusedScene:
        coord = CoordinateSystemSpec()
        fused = self._fusion.fuse(
            points_xyz=pts_xyz,
            seg_out=seg_out,
            detections=detections,
            coord_spec=coord,
        )
        if det_obbs is not None:
            fused = FusedScene(
                coord=fused.coord,
                points_xyz=fused.points_xyz,
                seg=fused.seg,
                detections=fused.detections,
                det_obbs=det_obbs,
            )
        return fused

    def __repr__(self) -> str:
        return (
            f"LightweightRealtimePipeline(camera={self._camera.source_name}, "
            f"voxel={self._voxel_size}, interval={self._proc_interval})"
        )
