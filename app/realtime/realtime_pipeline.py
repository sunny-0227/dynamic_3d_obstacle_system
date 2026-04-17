from __future__ import annotations

"""
实时流水线（双模式）

提供两种可独立使用的实时处理流水线：

  RealtimePipeline（已有）
  ─────────────────────────
  · 持有 IPointCloudCamera + FullPipeline（重量级离线模型）
  · 两种帧消费：
      read_raw()         — 仅读帧，不做分析；适合预览
      read_and_analyze() — 读帧 + 检测 + 分割 + 融合（调大模型）
  · 适合接入真实 OpenPCDet / MMDet3D；离线 demo 常用。

  LightweightRealtimePipeline（本次新增）
  ─────────────────────────────────────────
  · 持有 IPointCloudCamera + LightweightSegmentor + LightweightDetector
  · 不依赖 GPU 大模型；全纯 numpy/scipy/sklearn 实现
  · 两种帧消费：
      read_raw()         — 仅读帧，与 RealtimePipeline 一致
      read_and_analyze() — 读帧 → RANSAC 地面分割 → DBSCAN 聚类检测 →
                           FusedScene（含彩色点云 + OBB）
  · 特性：
      - 分割结果：地面（棕）/ 障碍物（红）/ 背景（灰）/ 噪点（深灰）
      - 检测结果：每个聚类生成 DetectionBox + Open3D OBB
      - 处理速度：128k 点约 10-30ms（取决于机器配置）
      - 可替换性：只需换 LightweightSegmentor / LightweightDetector 实现即可

  两个 pipeline 共用相同接口契约（read_raw / read_and_analyze / start / stop），
  可由 AppController 根据配置选择实例化哪个。

扩展说明：
  · 替换相机：修改 AppController._build_camera() 传入不同 IPointCloudCamera
  · 替换轻量算法：替换 LightweightSegmentor / LightweightDetector 的 config 或子类
  · 切换到重量级 pipeline：使用 RealtimePipeline + FullPipeline
"""

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
    """
    单帧处理结果（两个 pipeline 共用）。

    字段说明：
      frame     — 原始帧数据（CameraFrame）
      scene     — 融合场景（FusedScene），仅在 read_and_analyze() 时非 None
      clusters  — 轻量 pipeline 的聚类结果列表（RealtimePipeline 时为空）
      elapsed   — 本帧总处理耗时（秒）
    """

    frame: CameraFrame
    scene: Optional[FusedScene]
    clusters: List[ClusterResult] = field(default_factory=list)
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# ① RealtimePipeline（原有，保持不变）
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ② LightweightRealtimePipeline（新增）
# ---------------------------------------------------------------------------
class LightweightRealtimePipeline:
    """
    轻量级实时处理流水线：Camera → RANSAC 分割 → DBSCAN 聚类 → FusedScene

    与 RealtimePipeline 的区别：
      - 不依赖 FullPipeline（不需要 GPU / OpenPCDet / MMDet3D）
      - 使用 LightweightSegmentor + LightweightDetector
      - read_and_analyze() 的数据流：
          1. camera.get_next_frame() → CameraFrame（原始点云）
          2. LightweightSegmentor.segment() → 地面/障碍物/背景标签
          3. 提取 SEG_OBSTACLE 点作为聚类输入
          4. LightweightDetector.detect() → DetectionBox + ClusterResult
          5. _build_fused_scene() → FusedScene（供 SceneRenderer / GUI 使用）

    参数：
      camera      — 任意 IPointCloudCamera 实现
      seg_config  — 地面分割参数；None 使用默认
      det_config  — 聚类检测参数；None 使用默认
    """

    def __init__(
        self,
        camera: IPointCloudCamera,
        seg_config: Optional[GroundSegConfig] = None,
        det_config: Optional[ClusterConfig] = None,
    ) -> None:
        self._camera = camera
        self._segmentor = LightweightSegmentor(config=seg_config)
        self._detector = LightweightDetector(config=det_config)
        self._fusion = ResultFusion()

    # ------------------------------------------------------------------
    # 代理属性（与 RealtimePipeline 接口一致）
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return self._camera.source_name

    @property
    def is_running(self) -> bool:
        return self._camera.is_running

    @property
    def camera_info(self) -> CameraInfo:
        return self._camera.camera_info

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """启动相机，异常向上透传。"""
        logger.info("[LightweightRealtimePipeline] 启动相机：%s", self._camera.source_name)
        self._camera.start()
        logger.info(
            "[LightweightRealtimePipeline] 相机已启动，共 %d 帧可用",
            self.camera_info.total_files,
        )

    def stop(self) -> None:
        """停止相机（幂等）。"""
        logger.info("[LightweightRealtimePipeline] 停止相机：%s", self._camera.source_name)
        self._camera.stop()

    # ------------------------------------------------------------------
    # 帧消费模式（接口与 RealtimePipeline 完全一致）
    # ------------------------------------------------------------------

    def read_raw(self) -> RealtimeResult:
        """
        仅读取下一帧，不执行分割/检测。

        适用：预览模式、帧率测试。
        返回：RealtimeResult（scene=None, clusters=[]）
        """
        t0 = time.perf_counter()
        frame = self._camera.get_next_frame()
        elapsed = time.perf_counter() - t0

        logger.debug(
            "[LightweightRealtimePipeline] read_raw | 帧 #%d | 点数：%d | 耗时：%.1f ms",
            frame.frame_id,
            int(frame.points_xyz.shape[0]),
            elapsed * 1000,
        )
        return RealtimeResult(frame=frame, scene=None, clusters=[], elapsed=elapsed)

    def read_and_analyze(self) -> RealtimeResult:
        """
        读取一帧并执行轻量分割 + 聚类检测 + 融合。

        处理步骤（含耗时日志）：
          T0 → 读帧
          T1 → RANSAC 地面分割（生成语义标签 + 彩色点云）
          T2 → 提取障碍物候选点 → DBSCAN 聚类 → DetectionBox + OBB
          T3 → 组装 FusedScene

        返回：RealtimeResult（scene=FusedScene, clusters=ClusterResult列表）
        异常：camera / segmentor / detector 内部异常向上透传。
        """
        t0 = time.perf_counter()

        # ── 步骤 1：读帧 ─────────────────────────────────────────────
        frame = self._camera.get_next_frame()
        # 强制拷贝，防止后续算法意外修改 CameraFrame 内部数组
        pts_xyz = np.array(frame.points_xyz, dtype=np.float32)
        if pts_xyz.ndim == 1:
            pts_xyz = pts_xyz.reshape(-1, 3)
        pts_xyz = pts_xyz[:, :3]

        t1 = time.perf_counter()

        # ── 步骤 2：轻量语义分割 ─────────────────────────────────────
        seg_result, colored_pcd = self._segmentor.segment_with_colored_pcd(pts_xyz)
        seg_out = SegmentPipelineOutput(
            points_xyz=pts_xyz,
            seg=seg_result,
            colored_pcd=colored_pcd,
        )

        t2 = time.perf_counter()

        # ── 步骤 3：提取障碍物候选点，执行聚类检测 ────────────────────
        obstacle_mask = seg_result.labels == SEG_OBSTACLE
        obstacle_pts = pts_xyz[obstacle_mask]

        detections: List[DetectionBox] = []
        clusters: List[ClusterResult] = []
        det_obbs: Optional[list] = None

        if obstacle_pts.shape[0] >= self._detector._cfg.cluster_min_points:
            # detect_with_obbs 是 LightweightDetector 的公开接口，直接调用
            detections, clusters, det_obbs = self._detector.detect_with_obbs(
                obstacle_pts
            )
        else:
            logger.debug(
                "[LightweightRealtimePipeline] 障碍物候选点不足（%d），跳过聚类",
                obstacle_pts.shape[0],
            )

        t3 = time.perf_counter()

        # ── 步骤 4：组装 FusedScene ───────────────────────────────────
        scene = self._build_fused_scene(
            pts_xyz=pts_xyz,
            seg_out=seg_out,
            detections=detections,
            det_obbs=det_obbs,
        )

        elapsed = time.perf_counter() - t0
        n_pts = int(pts_xyz.shape[0])
        n_obs = int(obstacle_pts.shape[0])

        logger.debug(
            "[LightweightRealtimePipeline] read_and_analyze | 帧 #%d | 总点数：%d | "
            "障碍物点：%d | 聚类数：%d | "
            "读帧：%.1f ms | 分割：%.1f ms | 检测：%.1f ms | 总计：%.1f ms",
            frame.frame_id,
            n_pts,
            n_obs,
            len(detections),
            (t1 - t0) * 1000,
            (t2 - t1) * 1000,
            (t3 - t2) * 1000,
            elapsed * 1000,
        )

        return RealtimeResult(
            frame=frame,
            scene=scene,
            clusters=clusters,
            elapsed=elapsed,
        )

    # ------------------------------------------------------------------
    # 内部：构造 FusedScene
    # ------------------------------------------------------------------

    def _build_fused_scene(
        self,
        pts_xyz: np.ndarray,
        seg_out: SegmentPipelineOutput,
        detections: List[DetectionBox],
        det_obbs: Optional[list],
    ) -> FusedScene:
        """
        将分割结果 + 检测结果打包为 FusedScene。

        注意：det_obbs 已由 LightweightDetector 生成，直接注入 FusedScene，
        无需再经过 ResultFusion 的 BoxConverter（避免重复 convert 产生颜色丢失）。
        """
        coord = CoordinateSystemSpec()
        fused = self._fusion.fuse(
            points_xyz=pts_xyz,
            seg_out=seg_out,
            detections=detections,
            coord_spec=coord,
        )
        # 用轻量检测器已生成的彩色 OBB 覆盖 fusion 默认生成的 OBB
        if det_obbs is not None:
            # FusedScene 是 frozen dataclass，需创建新实例
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
            f"LightweightRealtimePipeline("
            f"camera={self._camera.source_name}, "
            f"running={self.is_running}"
            f")"
        )
