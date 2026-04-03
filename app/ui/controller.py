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
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal

from app.core.detector.openpcdet_detector import OpenPCDetDetector
from app.core.pipeline.detect_pipeline import DetectPipeline
from app.core.pipeline.full_pipeline import FullPipeline
from app.core.pipeline.segment_pipeline import SegmentPipeline
from app.core.segmentor.mmdet3d_segmentor import MMDet3DSegmentor, MMDet3DSegmentorConfig
from app.datasets.nuscenes_loader import NuScenesMiniLoader
from app.io.pointcloud_loader import (
    load_pointcloud,
    load_points_xyz_numpy,
    numpy_xyz_to_pointcloud,
)
from app.utils.logger import get_logger

logger = get_logger("ui.controller")


class _TaskThread(QThread):
    """
    一次性后台任务：重写 run()，不在子线程里跑 exec()。
    避免 moveToThread+QThread 默认事件循环与主线程 wait()/quit() 在 Windows 上偶发死锁。
    """

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

    def _store_loaded_pcd(
        self,
        pcd: o3d.geometry.PointCloud,
        *,
        current: Optional[Path] = None,
        clear_algo: bool = False,
    ) -> None:
        if current is not None:
            self.state.current_file = current
        if clear_algo:
            self.state.last_det = None
            self.state.last_seg = None
            self.state.last_scene = None
        self.state.loaded_pcd = pcd

    def _notify_pointcloud_loaded(self, success_log: str, *, open_preview: bool) -> None:
        self.sig_log.emit(success_log)
        self.sig_status.emit("点云已加载")
        if open_preview:
            self.sig_request_render.emit("raw")

    def _load_pcd_on_main_thread(
        self,
        path: Path,
        *,
        current: Optional[Path] = None,
        clear_algo: bool = False,
        success_log: Optional[str] = None,
    ) -> bool:
        """在主线程用 Open3D 读 .pcd；成功返回 True 并可选打开预览。"""
        self.sig_busy.emit(True)
        ok = False
        try:
            pcd = load_pointcloud(path)
            self._store_loaded_pcd(pcd, current=current, clear_algo=clear_algo)
            msg = success_log if success_log is not None else f"加载成功：点数 {len(pcd.points):,}"
            self._notify_pointcloud_loaded(msg, open_preview=True)
            ok = True
        except Exception as e:
            logger.error("主线程加载 .pcd 失败: %s", e, exc_info=True)
            self.sig_error.emit("加载失败", str(e))
            self.sig_status.emit("加载失败")
        finally:
            self.sig_busy.emit(False)
            self._emit_state()
        return ok

    def _run_in_thread(self, fn: Callable[[], Any], on_done: Callable[[Any], None]) -> None:
        if self._thread is not None:
            # 简化：不支持并行任务，避免状态竞争
            self.sig_log.emit("已有任务正在运行，请稍候…")
            return

        self.sig_busy.emit(True)
        self._thread = _TaskThread(fn)
        self._thread.sig_done.connect(
            lambda res: self._on_thread_done(res, on_done),
            Qt.QueuedConnection,
        )
        self._thread.sig_error.connect(self._on_thread_error, Qt.QueuedConnection)
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
            t = self._thread
            self._thread = None
            # 须在槽内 wait 后再 deleteLater：避免 finished 早于 sig_done 导致对象被提前释放
            if t.isRunning() and not t.wait(5000):
                logger.warning("后台线程 5s 内未结束，若界面异常可重启程序")
            t.deleteLater()
        # 先解除 busy，再根据最新状态刷新按钮（避免 restore 覆盖 on_done 里启用的控件）
        self.sig_busy.emit(False)
        self._emit_state()

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

        # .pcd 主线程 Open3D；.bin 子线程仅 numpy，主线程再建 PointCloud（避免 Windows 子线程卡死）
        if p.suffix.lower() == ".pcd":
            self._load_pcd_on_main_thread(p)
            return

        def job():
            return load_points_xyz_numpy(p)

        def done(xyz):
            self._store_loaded_pcd(numpy_xyz_to_pointcloud(xyz))
            self._notify_pointcloud_loaded(
                f"加载成功：点数 {len(self.state.loaded_pcd.points):,}",
                open_preview=True,
            )

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

        try:
            rec = loader.get_frame_record(idx)
        except Exception as e:
            self.sig_error.emit("获取帧失败", str(e))
            return

        if not rec.lidar_path.is_file():
            self.sig_error.emit("文件缺失", str(rec.lidar_path))
            return

        lp = rec.lidar_path
        success_log = f"已加载 nuScenes 帧：{idx+1}/{rec.frame_count} | {rec.lidar_path.name}"
        if lp.suffix.lower() == ".pcd":
            self.sig_status.emit("加载 nuScenes 点云中…")
            self._load_pcd_on_main_thread(
                lp, current=lp, clear_algo=True, success_log=success_log
            )
            return

        def job():
            return rec, load_points_xyz_numpy(lp)

        def done(res):
            rec2, xyz = res
            pcd = numpy_xyz_to_pointcloud(xyz)
            self._store_loaded_pcd(pcd, current=rec2.lidar_path, clear_algo=True)
            self._notify_pointcloud_loaded(success_log, open_preview=True)

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
            cf = self.state.current_file
            if cf.suffix.lower() == ".pcd":
                self.sig_log.emit("一键运行：点云未加载，在主线程加载 .pcd …")
                self.sig_busy.emit(True)
                try:
                    self._store_loaded_pcd(load_pointcloud(cf))
                    self._emit_state()
                    # 须等 busy 收尾后再启动一键，否则 _run_in_thread 会判「已有任务」
                    QTimer.singleShot(0, self.run_full)
                except Exception as e:
                    self.sig_error.emit("加载失败", str(e))
                finally:
                    self.sig_busy.emit(False)
                    self._emit_state()
                return

            self.sig_log.emit("一键运行：点云未加载，先自动加载…")

            def job_load():
                return load_points_xyz_numpy(cf)

            def done_load(xyz):
                self._store_loaded_pcd(numpy_xyz_to_pointcloud(xyz))
                self._emit_state()
                QTimer.singleShot(0, self.run_full)

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

