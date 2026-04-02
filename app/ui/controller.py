"""
UI 控制器（里程碑 6）

职责：
  - 管理应用状态（当前文件、点云、nuScenes loader、最近结果）
  - 在后台线程执行耗时任务（加载点云/连接数据集/检测/分割/一键运行）
  - 在主线程触发 Open3D 显示（渲染仍会阻塞，但计算尽量异步）
  - 统一异常提示与日志输出
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import open3d as o3d
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from app.core.detector.openpcdet_detector import OpenPCDetDetector
from app.core.pipeline.detect_pipeline import DetectPipeline
from app.core.pipeline.full_pipeline import FullPipeline
from app.core.pipeline.segment_pipeline import SegmentPipeline
from app.core.segmentor.mmdet3d_segmentor import MMDet3DSegmentor, MMDet3DSegmentorConfig
from app.datasets.nuscenes_loader import NuScenesMiniLoader
from app.io.pointcloud_loader import load_pointcloud
from app.utils.logger import get_logger

logger = get_logger("ui.controller")


class _Worker(QObject):
    sig_done = pyqtSignal(object)
    sig_error = pyqtSignal(str, str)

    def __init__(self, fn: Callable[[], Any]):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            res = self._fn()
            self.sig_done.emit(res)
        except Exception as e:
            logger.error("后台任务失败: %s", e, exc_info=True)
            self.sig_error.emit("任务失败", str(e))


@dataclass
class AppState:
    current_file: Optional[Path] = None
    loaded_pcd: Optional[o3d.geometry.PointCloud] = None

    nusc_root: Optional[Path] = None
    nusc_loader: Optional[NuScenesMiniLoader] = None

    last_det = None
    last_seg = None
    last_scene = None


class AppController(QObject):
    sig_log = pyqtSignal(str)
    sig_status = pyqtSignal(str)
    sig_error = pyqtSignal(str, str)  # title, message
    sig_state = pyqtSignal(object)  # AppState
    sig_busy = pyqtSignal(bool)
    sig_request_render = pyqtSignal(str)  # "raw" / "seg" / "fusion"

    def __init__(self, config: dict):
        super().__init__()
        self._config = config or {}
        self._project_root = Path(__file__).resolve().parent.parent.parent
        self.state = AppState()

        # pipelines（懒加载）
        self._detector_pipeline: Optional[DetectPipeline] = None
        self._segment_pipeline: Optional[SegmentPipeline] = None
        self._full_pipeline: Optional[FullPipeline] = None

        # 线程
        self._thread: Optional[QThread] = None

    # -----------------------------
    # 内部工具
    # -----------------------------
    def _emit_state(self) -> None:
        self.sig_state.emit(self.state)

    def _run_in_thread(self, fn: Callable[[], Any], on_done: Callable[[Any], None]) -> None:
        if self._thread is not None:
            # 简化：不支持并行任务，避免状态竞争
            self.sig_log.emit("已有任务正在运行，请稍候…")
            return

        self.sig_busy.emit(True)
        self._thread = QThread()
        worker = _Worker(fn)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.sig_done.connect(lambda res: self._on_thread_done(res, on_done))
        worker.sig_error.connect(self._on_thread_error)
        worker.sig_done.connect(worker.deleteLater)
        worker.sig_error.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_thread_done(self, res: Any, on_done: Callable[[Any], None]) -> None:
        try:
            on_done(res)
        finally:
            self._finish_thread()

    def _on_thread_error(self, title: str, msg: str) -> None:
        self.sig_error.emit(title, msg)
        self._finish_thread()

    def _finish_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
        self.sig_busy.emit(False)

    def _ensure_pipelines(self) -> None:
        if self._detector_pipeline is None:
            det_cfg = self._config.get("detector", {})
            score_threshold = det_cfg.get("score_threshold", 0.1)
            num_boxes_fake = det_cfg.get("num_boxes_fake", 3)
            class_names = det_cfg.get("class_names", ["car", "pedestrian", "cyclist"])
            openpcdet_cfg = det_cfg.get("openpcdet", {})

            def _resolve(p: str) -> Optional[Path]:
                p = str(p or "").strip()
                if not p:
                    return None
                pp = Path(p)
                if not pp.is_absolute():
                    pp = self._project_root / pp
                return pp

            detector = OpenPCDetDetector(
                model_cfg_path=_resolve(openpcdet_cfg.get("model_cfg", "")),
                checkpoint_path=_resolve(openpcdet_cfg.get("checkpoint_path", "")),
                device=openpcdet_cfg.get("device", "cpu"),
                score_threshold=score_threshold,
                num_boxes_fake=num_boxes_fake,
                class_names=class_names,
            )
            self._detector_pipeline = DetectPipeline(detector=detector)

        if self._segment_pipeline is None:
            seg_cfg = self._config.get("segmentor", {})
            mmd = seg_cfg.get("mmdet3d", {})
            palette = seg_cfg.get("palette", None) or self._config.get("fake_segmentor", {}).get("class_colors", None)
            seg = MMDet3DSegmentor(
                MMDet3DSegmentorConfig(
                    config_file=str(mmd.get("config_file", "") or ""),
                    checkpoint_file=str(mmd.get("checkpoint_file", "") or ""),
                    device=str(mmd.get("device", "cpu") or "cpu"),
                    num_classes=int(seg_cfg.get("num_classes", 4)),
                    class_names=seg_cfg.get("class_names", None),
                    palette=palette,
                )
            )
            self._segment_pipeline = SegmentPipeline(segmentor=seg)

        if self._full_pipeline is None:
            self._full_pipeline = FullPipeline(
                detect_pipeline=self._detector_pipeline,
                segment_pipeline=self._segment_pipeline,
            )

    # -----------------------------
    # 对外动作：文件/数据集
    # -----------------------------
    def set_current_file(self, path: Optional[Path]) -> None:
        self.state.current_file = path
        self.state.loaded_pcd = None
        self.state.last_det = None
        self.state.last_seg = None
        self.state.last_scene = None
        self._emit_state()

    def load_current_file_pointcloud(self) -> None:
        if self.state.current_file is None:
            self.sig_error.emit("提示", "请先选择点云文件")
            return

        p = self.state.current_file
        if not p.is_file():
            self.sig_error.emit("文件不存在", str(p))
            return

        self.sig_status.emit("加载点云中…")
        self.sig_log.emit(f"正在加载点云：{p}")

        def job():
            pcd = load_pointcloud(p)
            return pcd

        def done(pcd):
            self.state.loaded_pcd = pcd
            n = len(pcd.points)
            self.sig_log.emit(f"加载成功：点数 {n:,}")
            self.sig_status.emit("点云已加载")
            self._emit_state()

        self._run_in_thread(job, done)

    def set_nusc_root(self, root: Optional[Path]) -> None:
        self.state.nusc_root = root
        self.state.nusc_loader = None
        self._emit_state()

    def connect_nusc(self) -> None:
        if self.state.nusc_root is None:
            self.sig_error.emit("提示", "请先选择 nuScenes 根目录")
            return

        ver = self._config.get("nuscenes", {}).get("version", "v1.0-mini")
        root = self.state.nusc_root
        self.sig_status.emit("连接 nuScenes 中…")
        self.sig_log.emit(f"连接 nuScenes：root={root} version={ver}")

        def job():
            loader = NuScenesMiniLoader(root, version=ver)
            loader.connect()
            return loader

        def done(loader: NuScenesMiniLoader):
            self.state.nusc_loader = loader
            mode = loader.mode
            self.sig_log.emit(f"nuScenes 连接成功（mode={mode}）")
            self.sig_status.emit("nuScenes 已连接")
            self._emit_state()

        self._run_in_thread(job, done)

    def set_nusc_navigation(self, mode: str, scene_token: Optional[str] = None) -> None:
        loader = self.state.nusc_loader
        if loader is None or not loader.is_connected:
            return
        try:
            if mode == "global":
                loader.set_navigation("global")
            else:
                loader.set_navigation("scene", scene_token=scene_token)
            self._emit_state()
        except Exception as e:
            self.sig_error.emit("导航失败", str(e))

    def load_nusc_frame(self, frame_index: int) -> None:
        loader = self.state.nusc_loader
        if loader is None or not loader.is_connected:
            self.sig_error.emit("提示", "请先加载 nuScenes 数据集")
            return
        idx = int(frame_index)

        def job():
            rec = loader.get_frame_record(idx)
            if not rec.lidar_path.is_file():
                raise FileNotFoundError(str(rec.lidar_path))
            pcd = load_pointcloud(rec.lidar_path)
            return rec, pcd

        def done(res):
            rec, pcd = res
            self.state.current_file = rec.lidar_path
            self.state.loaded_pcd = pcd
            self.state.last_det = None
            self.state.last_seg = None
            self.state.last_scene = None
            self.sig_log.emit(f"已加载 nuScenes 帧：{idx+1}/{rec.frame_count} | {rec.lidar_path.name}")
            self.sig_status.emit("点云已加载")
            self._emit_state()

        self.sig_status.emit("加载 nuScenes 点云中…")
        self._run_in_thread(job, done)

    # -----------------------------
    # 对外动作：算法
    # -----------------------------
    def run_detect(self) -> None:
        if self.state.loaded_pcd is None:
            self.sig_error.emit("提示", "请先加载点云")
            return
        self._ensure_pipelines()
        pts = np.asarray(self.state.loaded_pcd.points, dtype=np.float32)
        self.sig_status.emit("检测中…")
        self.sig_log.emit("开始执行检测…")

        def job():
            return self._detector_pipeline.run(pts)

        def done(dets):
            self.state.last_det = dets
            self.sig_log.emit(f"检测完成：检测框 {len(dets)} 个")
            self.sig_status.emit("检测完成")
            self._emit_state()
            self.sig_request_render.emit("fusion")

        self._run_in_thread(job, done)

    def run_segment(self) -> None:
        if self.state.loaded_pcd is None:
            self.sig_error.emit("提示", "请先加载点云")
            return
        self._ensure_pipelines()
        pts = np.asarray(self.state.loaded_pcd.points, dtype=np.float32)
        self.sig_status.emit("分割中…")
        self.sig_log.emit("开始执行分割…")

        def job():
            return self._segment_pipeline.run(pts)

        def done(seg_out):
            self.state.last_seg = seg_out
            self.sig_log.emit("分割完成")
            self.sig_status.emit("分割完成")
            self._emit_state()
            self.sig_request_render.emit("seg")

        self._run_in_thread(job, done)

    def run_full(self) -> None:
        # 一键运行：若未加载但有 current_file 则先加载再跑（单文件演示更顺畅）
        if self.state.loaded_pcd is None and self.state.current_file is not None and self.state.current_file.is_file():
            self.sig_log.emit("一键运行：点云未加载，先自动加载…")

            def job_load():
                return load_pointcloud(self.state.current_file)

            def done_load(pcd):
                self.state.loaded_pcd = pcd
                self._emit_state()
                self.run_full()

            self._run_in_thread(job_load, done_load)
            return

        if self.state.loaded_pcd is None:
            self.sig_error.emit("提示", "请先选择/加载点云或加载 nuScenes 当前帧点云")
            return

        self._ensure_pipelines()
        pts = np.asarray(self.state.loaded_pcd.points, dtype=np.float32)
        self.sig_status.emit("一键运行中…")
        self.sig_log.emit("开始一键运行（分割+检测+融合）…")

        def job():
            return self._full_pipeline.run(pts)

        def done(scene):
            self.state.last_scene = scene
            self.sig_log.emit("一键运行完成（已生成融合场景）")
            self.sig_status.emit("一键运行完成")
            self._emit_state()
            self.sig_request_render.emit("fusion")

        self._run_in_thread(job, done)

    def clear_results(self) -> None:
        self.state.last_det = None
        self.state.last_seg = None
        self.state.last_scene = None
        self.sig_log.emit("已清空结果")
        self.sig_status.emit("就绪")
        self._emit_state()

