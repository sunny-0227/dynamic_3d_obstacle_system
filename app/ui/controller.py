"""
UI 控制器（里程碑 6）

主界面在「一键分析」等按钮上采用更严格的前置条件（由主窗口 set_action_buttons_state 控制）；
控制器内仍保留单文件自动加载逻辑，供非 GUI 调用或后续放宽策略时使用。

职责：
  - 管理应用状态（当前文件、点云、nuScenes loader、最近结果）
  - 在后台线程执行耗时任务（加载点云/连接数据集/检测/分割/一键分析）
  - 在主线程触发 Open3D 显示（渲染仍会阻塞，但计算尽量异步）
  - 统一异常提示与日志输出
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

WorkflowMode = Literal["none", "single_file", "nuscenes"]
RuntimeMode = Literal["offline", "realtime"]

import numpy as np
import open3d as o3d
import time
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal

from app.core.detector.openpcdet_detector import OpenPCDetDetector
from app.core.detector.openpcdet_json_detector import OpenPCDetJsonDetector
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
from app.realtime.mock_camera import MockCamera
from app.realtime.realsense_camera import RealSenseCamera
from app.realtime.realtime_pipeline import RealtimePipeline
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




class _RealtimeThread(QThread):
    """实时循环线程：读取点云帧，可选执行实时分析。"""

    sig_result = pyqtSignal(object, float)  # RealtimeResult, fps
    sig_error = pyqtSignal(str, str)

    def __init__(self, rt_pipeline: RealtimePipeline, analyze: bool, target_fps: float):
        super().__init__()
        self._rt = rt_pipeline
        self._analyze = bool(analyze)
        self._running = True
        self._dt = 1.0 / max(float(target_fps), 1.0)

    def request_stop(self) -> None:
        self._running = False

    def run(self) -> None:
        prev_t = time.perf_counter()
        while self._running:
            t0 = time.perf_counter()
            try:
                if self._analyze:
                    res = self._rt.read_and_analyze()
                else:
                    res = self._rt.read_raw()
                now = time.perf_counter()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                self.sig_result.emit(res, float(fps))
            except Exception as e:
                self.sig_error.emit("实时模式失败", str(e))
                break
            elapsed = time.perf_counter() - t0
            sleep_s = self._dt - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)


@dataclass
class AppState:
    """workflow：离线流程；runtime_mode：离线/实时模式切换。"""

    runtime_mode: RuntimeMode = "offline"
    workflow: WorkflowMode = "none"
    current_file: Optional[Path] = None
    loaded_pcd: Optional[o3d.geometry.PointCloud] = None

    nusc_root: Optional[Path] = None
    nusc_loader: Optional[NuScenesMiniLoader] = None

    last_det = None
    last_seg = None
    last_scene = None

    realtime_stream_dir: Optional[Path] = None
    realtime_running: bool = False
    realtime_analyzing: bool = False
    realtime_fps: float = 0.0
    realtime_points: int = 0
    realtime_source: str = "Mock"


class AppController(QObject):
    sig_log = pyqtSignal(str)
    sig_status = pyqtSignal(str)
    sig_error = pyqtSignal(str, str)  # title, message
    sig_state = pyqtSignal(object)  # AppState
    sig_busy = pyqtSignal(bool)
    sig_request_render = pyqtSignal(str)  # "raw" / "seg" / "fusion"
    # 实时分析帧结果：携带 FusedScene，驱动实时 Open3D 窗口刷新
    sig_realtime_frame = pyqtSignal(object)  # FusedScene

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
        self._rt_thread: Optional[_RealtimeThread] = None
        self._rt_pipeline: Optional[RealtimePipeline] = None

    # -----------------------------
    # 内部工具
    # -----------------------------
    def _emit_state(self) -> None:
        self.sig_state.emit(self.state)

    def _reject_empty_pcd(self) -> bool:
        """若当前点云为空则清空并提示。返回 True 表示可继续。"""
        pcd = self.state.loaded_pcd
        if pcd is None or len(pcd.points) == 0:
            self.state.loaded_pcd = None
            self.sig_error.emit("点云无效", "点云为空（0 个点），无法继续后续操作。")
            self.sig_status.emit("点云无效（空）")
            return False
        return True

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

    def _notify_pointcloud_loaded(self, success_log: str, *, open_preview: bool) -> bool:
        if not self._reject_empty_pcd():
            return False
        self.sig_log.emit(success_log)
        self.sig_status.emit("点云已加载")
        if open_preview:
            self.sig_request_render.emit("raw")
        return True

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
            ok = self._notify_pointcloud_loaded(msg, open_preview=True)
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


    def _stop_realtime_thread(self) -> None:
        if self._rt_thread is None:
            return
        t = self._rt_thread
        self._rt_thread = None
        t.request_stop()
        if t.isRunning():
            t.wait(3000)
        t.deleteLater()

    def _ensure_pipelines(self) -> None:
        if self._detector_pipeline is None:
            det_cfg   = self._config.get("detector", {})
            score_threshold = det_cfg.get("score_threshold", 0.1)
            num_boxes_fake  = det_cfg.get("num_boxes_fake", 3)
            class_names     = det_cfg.get("class_names", ["car", "pedestrian", "cyclist"])
            wsl_cfg   = det_cfg.get("openpcdet_wsl", {})

            # ── 优先：WSL2 subprocess 真实推理 ────────────────────────
            # enable_wsl=true 且关键路径已填写时，使用 OpenPCDetJsonDetector；
            # 否则静默降级为旧版 OpenPCDetDetector（内置 fake）。
            enable_wsl    = bool(wsl_cfg.get("enable_wsl", False))
            cfg_file_wsl  = str(wsl_cfg.get("cfg_file",      "") or "").strip()
            ckpt_file_wsl = str(wsl_cfg.get("ckpt_file",     "") or "").strip()
            infer_script  = str(wsl_cfg.get("infer_script",  "") or "").strip()
            conda_env     = str(wsl_cfg.get("conda_env",     "openpcdet")).strip()
            ext           = str(wsl_cfg.get("ext",           ".bin")).strip()
            timeout_s     = int(wsl_cfg.get("wsl_timeout_s", 120))
            tmp_dir_str   = str(wsl_cfg.get("tmp_dir",       "") or "").strip()
            tmp_dir       = Path(tmp_dir_str) if tmp_dir_str else None

            # 判断 WSL 配置是否完整（三个路径均非空才真正启用）
            wsl_paths_ok = bool(cfg_file_wsl and ckpt_file_wsl and infer_script)

            if enable_wsl and wsl_paths_ok:
                logger.info(
                    "[Controller] 检测器：OpenPCDetJsonDetector（WSL 真实推理模式）"
                )
                detector = OpenPCDetJsonDetector(
                    cfg_file      = cfg_file_wsl,
                    ckpt_file     = ckpt_file_wsl,
                    infer_script  = infer_script,
                    conda_env     = conda_env,
                    ext           = ext,
                    score_threshold = score_threshold,
                    class_names   = class_names,
                    wsl_timeout_s = timeout_s,
                    num_boxes_fake= num_boxes_fake,
                    tmp_dir       = tmp_dir,
                    enable_wsl    = True,
                )
            else:
                # ── 降级：旧版 OpenPCDetDetector（内置 fake）─────────
                if enable_wsl and not wsl_paths_ok:
                    logger.warning(
                        "[Controller] enable_wsl=true 但 cfg_file/ckpt_file/infer_script "
                        "未全部配置，降级为模拟检测。"
                        "请在 config/settings.yaml [detector.openpcdet_wsl] 中填写完整路径。"
                    )
                else:
                    logger.info(
                        "[Controller] 检测器：OpenPCDetDetector（模拟检测 / 占位模式）"
                    )

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
                    model_cfg_path  = _resolve(openpcdet_cfg.get("model_cfg", "")),
                    checkpoint_path = _resolve(openpcdet_cfg.get("checkpoint_path", "")),
                    device          = openpcdet_cfg.get("device", "cpu"),
                    score_threshold = score_threshold,
                    num_boxes_fake  = num_boxes_fake,
                    class_names     = class_names,
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
        if path is not None:
            self.state.workflow = "single_file"
            self.state.nusc_loader = None
            self.state.nusc_root = None
            self.sig_log.emit(
                f"[模式] 单文件点云 | 已选文件路径: {path.resolve() if path.is_absolute() else path}"
            )
        else:
            self.state.workflow = "none"
        self._emit_state()

    def disconnect_nusc(self) -> None:
        """断开 nuScenes 连接并清空相关状态，用于切换回单文件模式。"""
        self.state.nusc_loader = None
        self.state.nusc_root = None
        self.state.current_file = None
        self.state.loaded_pcd = None
        self.state.last_det = None
        self.state.last_seg = None
        self.state.last_scene = None
        self.state.workflow = "none"
        self.sig_log.emit("[模式] 已断开 nuScenes，工作流重置为未选择")
        self.sig_status.emit("就绪")
        self._emit_state()

    def load_current_file_pointcloud(self) -> None:
        if self.state.workflow == "nuscenes":
            self.sig_error.emit(
                "当前为 nuScenes 模式",
                "请使用「加载当前帧点云」载入数据。\n"
                "若需改用单文件点云，请先选择「选择点云文件」并确认切换到单文件模式。",
            )
            return
        if self.state.current_file is None:
            self.sig_error.emit("提示", "请先选择点云文件")
            return

        p = self.state.current_file
        if not p.is_file():
            self.sig_error.emit("文件不存在", str(p))
            return

        self.sig_status.emit("加载点云中…")
        self.sig_log.emit(f"[单文件] 正在加载点云：{p}")

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
        if root is not None:
            self.sig_log.emit(f"[nuScenes] 已选择数据集根目录（尚未连接）: {root.as_posix()}")
        self._emit_state()

    def connect_nusc(self, nav_mode: str = "global", scene_token: Optional[str] = None) -> None:
        """
        连接 nuScenes 数据集。

        nav_mode / scene_token 来自界面「导航方式」下拉框（未改则默认 global）。
        connect() 内部会先建立默认全局帧列表；此处再按用户选择覆盖为 scene（若适用）。
        """
        if self.state.nusc_root is None:
            self.sig_error.emit("提示", "请先选择 nuScenes 根目录")
            return

        ver = self._config.get("nuscenes", {}).get("version", "v1.0-mini")
        root = self.state.nusc_root
        self.sig_status.emit("连接 nuScenes 中…")
        self.sig_log.emit(f"连接 nuScenes：root={root.as_posix()} version={ver}")

        def job():
            loader = NuScenesMiniLoader(root, version=ver)
            loader.connect()
            return loader

        def done(loader: NuScenesMiniLoader):
            self.state.nusc_loader = loader
            self.state.workflow = "nuscenes"
            self.state.current_file = None
            self.state.loaded_pcd = None
            self.state.last_det = None
            self.state.last_seg = None
            self.state.last_scene = None
            root = self.state.nusc_root
            try:
                self._apply_initial_nusc_navigation(loader, nav_mode, scene_token)
            except Exception as e:
                logger.warning("按界面选择初始化导航失败，回退全数据集: %s", e, exc_info=True)
                try:
                    loader.set_navigation("global")
                except Exception:
                    pass
                self.sig_log.emit(
                    f"[nuScenes] 导航初始化异常，已回退为默认：{loader.navigation_display_zh()} | {e}"
                )

            nav_label = loader.navigation_display_zh()
            n_frames = loader.frame_count
            self.sig_log.emit(
                f"[模式] nuScenes 数据集 | 根目录: {root.as_posix()}\n"
                f"| 数据模式: {loader.mode_display_zh()}（内部 mode={loader.mode}）\n"
                f"| 导航方式: {nav_label} | 帧数: {n_frames}\n"
                "| 可直接调整帧索引并点击「加载当前帧点云」（可随时在下拉框切换导航方式）。"
            )
            # 顶栏/底栏「当前状态」「执行」由 sig_status → 主窗口 _on_controller_status 更新
            self.sig_status.emit(f"nuScenes 已连接（{nav_label}）｜请加载当前帧点云")
            self._emit_state()

        self._run_in_thread(job, done)

    def _apply_initial_nusc_navigation(
        self,
        loader: NuScenesMiniLoader,
        nav_mode: str,
        scene_token: Optional[str],
    ) -> None:
        """连接成功后：默认 global；若界面为 scene 则切换到指定或首个场景。"""
        mode = str(nav_mode or "global").strip().lower()
        if mode not in ("global", "scene"):
            mode = "global"

        if mode == "global":
            loader.set_navigation("global")
            return

        tok = str(scene_token).strip() if scene_token is not None else ""
        if not tok:
            summaries = loader.get_scene_summaries()
            if summaries:
                tok = str(summaries[0]["token"])
        if not tok:
            loader.set_navigation("global")
            self.sig_log.emit(
                "[nuScenes] 按场景导航但无可用场景 token，已使用默认：全数据集（sample 表顺序，非时间序）"
            )
            return
        loader.set_navigation("scene", scene_token=tok)

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
        if loader is not None:
            self.sig_log.emit(
                f"[nuScenes] 场景: {rec.scene_name} | 帧索引: {idx}（0 基）/ 共 {rec.frame_count} 帧 | "
                f"LiDAR: {lp.as_posix()}\n"
                f"| 导航: {loader.navigation_display_zh()} | 数据模式: {loader.mode_display_zh()}"
            )
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
        if self.state.loaded_pcd is None or len(self.state.loaded_pcd.points) == 0:
            self.sig_error.emit("提示", "请先加载非空点云后再执行检测。")
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
        if self.state.loaded_pcd is None or len(self.state.loaded_pcd.points) == 0:
            self.sig_error.emit("提示", "请先加载非空点云后再执行分割。")
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
        # nuScenes：必须先加载当前帧点云（禁止静默用磁盘路径自动加载）
        if self.state.workflow == "nuscenes":
            if self.state.loaded_pcd is None or len(self.state.loaded_pcd.points) == 0:
                self.sig_error.emit(
                    "无法一键分析",
                    "请先点击「加载当前帧点云」载入当前帧，并确认点云非空后再执行一键分析。",
                )
                return
            self._run_full_pipeline()
            return

        # 单文件：若未加载但有 current_file 则先加载再跑（单文件流程更顺畅）
        if self.state.loaded_pcd is None and self.state.current_file is not None and self.state.current_file.is_file():
            cf = self.state.current_file
            if cf.suffix.lower() == ".pcd":
                self.sig_log.emit("一键分析：点云未加载，在主线程加载 .pcd …")
                self.sig_busy.emit(True)
                try:
                    self._store_loaded_pcd(load_pointcloud(cf))
                    self._emit_state()
                    # 须等 busy 收尾后再启动一键分析，否则 _run_in_thread 会判「已有任务」
                    QTimer.singleShot(0, self.run_full)
                except Exception as e:
                    self.sig_error.emit("加载失败", str(e))
                finally:
                    self.sig_busy.emit(False)
                    self._emit_state()
                return

            self.sig_log.emit("一键分析：点云未加载，先自动加载…")

            def job_load():
                return load_points_xyz_numpy(cf)

            def done_load(xyz):
                self._store_loaded_pcd(numpy_xyz_to_pointcloud(xyz))
                self._emit_state()
                QTimer.singleShot(0, self.run_full)

            self._run_in_thread(job_load, done_load)
            return

        if self.state.loaded_pcd is None:
            self.sig_error.emit(
                "无法一键分析",
                "请先加载点云：单文件模式请点击「加载点云」；nuScenes 模式请先连接数据集并「加载当前帧点云」。",
            )
            return

        if len(self.state.loaded_pcd.points) == 0:
            self.sig_error.emit("无法一键分析", "当前点云为空，请重新加载有效点云。")
            return

        self._run_full_pipeline()

    def _run_full_pipeline(self) -> None:
        self._ensure_pipelines()
        pts = np.asarray(self.state.loaded_pcd.points, dtype=np.float32)
        self.sig_status.emit("一键分析中…")
        self.sig_log.emit("开始一键分析（分割+检测+融合）…")

        def job():
            return self._full_pipeline.run(pts)

        def done(scene):
            self.state.last_scene = scene
            self.sig_log.emit("一键分析完成（已生成融合场景）")
            self.sig_status.emit("一键分析完成")
            self._emit_state()
            self.sig_request_render.emit("fusion")

        self._run_in_thread(job, done)

    # -----------------------------
    # 对外动作：实时模式（Mock Camera）
    # -----------------------------
    def set_runtime_mode(self, mode: RuntimeMode) -> None:
        self.state.runtime_mode = "realtime" if mode == "realtime" else "offline"
        self._emit_state()

    def set_realtime_stream_dir(self, path: Optional[Path]) -> None:
        self.state.realtime_stream_dir = path
        self._emit_state()

    def _build_camera(self):
        """
        根据配置构建相机实例。

        配置项（config.realtime.camera_type）：
          "mock"       — MockCamera（默认，不需要硬件）
          "realsense"  — RealSenseCamera（需要 pyrealsense2 和硬件）
        """
        rt_cfg = self._config.get("realtime", {})
        camera_type = str(rt_cfg.get("camera_type", "mock")).lower().strip()

        if camera_type == "realsense":
            cam = RealSenseCamera(
                width=int(rt_cfg.get("width", 640)),
                height=int(rt_cfg.get("height", 480)),
                fps=int(rt_cfg.get("fps", 30)),
                serial_number=str(rt_cfg.get("serial_number", "")),
                align_to_color=bool(rt_cfg.get("align_to_color", False)),
                min_depth_m=float(rt_cfg.get("min_depth_m", 0.1)),
                max_depth_m=float(rt_cfg.get("max_depth_m", 10.0)),
                timeout_ms=int(rt_cfg.get("timeout_ms", 2000)),
            )
            self.sig_log.emit(
                f"[实时] 相机类型：RealSense | "
                f"{rt_cfg.get('width', 640)}x{rt_cfg.get('height', 480)} "
                f"@ {rt_cfg.get('fps', 30)}fps"
            )
            return cam

        # 默认 Mock Camera
        stream_dir = self.state.realtime_stream_dir or (self._project_root / "data")
        if not stream_dir.is_dir():
            raise FileNotFoundError(
                f"Mock Camera 目录不存在，请先选择包含 .bin 或 .pcd 文件的目录。\n"
                f"当前路径：{stream_dir}"
            )
        cam = MockCamera(
            stream_dir,
            loop=bool(rt_cfg.get("mock_loop", True)),
            target_fps=float(rt_cfg.get("mock_fps", 5.0)),
        )
        self.sig_log.emit(f"[实时] 相机类型：Mock Camera | 目录：{stream_dir.as_posix()}")
        return cam

    def start_realtime_mode(self) -> None:
        if self.state.realtime_running:
            return
        # 离线任务正在运行时不允许启动实时模式，避免状态竞争
        if self._thread is not None:
            self.sig_error.emit("提示", "当前有离线任务正在运行，请等待任务完成后再启动实时模式。")
            return

        # 构建相机（Mock 或 RealSense，由配置决定）
        try:
            cam = self._build_camera()
        except Exception as e:
            self.sig_error.emit("实时模式启动失败", str(e))
            return

        self._ensure_pipelines()
        self._rt_pipeline = RealtimePipeline(cam, self._full_pipeline)
        try:
            self._rt_pipeline.start()
        except Exception as e:
            self.sig_error.emit("启动实时模式失败", str(e))
            self._rt_pipeline = None
            return

        rt_cfg = self._config.get("realtime", {})
        raw_fps = float(rt_cfg.get("raw_fps", 5.0))

        self.state.runtime_mode = "realtime"
        self.state.realtime_running = True
        self.state.realtime_analyzing = False
        self.state.realtime_source = self._rt_pipeline.source_name
        self.sig_status.emit("实时模式已启动")
        self.sig_log.emit(
            f"[实时] 已启动实时模式 | 数据源={self.state.realtime_source} | 预览帧率={raw_fps:.1f}fps"
        )

        self._rt_thread = _RealtimeThread(self._rt_pipeline, analyze=False, target_fps=raw_fps)
        self._rt_thread.sig_result.connect(self._on_realtime_result, Qt.QueuedConnection)
        self._rt_thread.sig_error.connect(self._on_realtime_error, Qt.QueuedConnection)
        self._rt_thread.start()
        self._emit_state()

    def stop_realtime_mode(self) -> None:
        self._stop_realtime_thread()
        if self._rt_pipeline is not None:
            try:
                self._rt_pipeline.stop()
            except Exception:
                pass
            self._rt_pipeline = None
        self.state.realtime_running = False
        self.state.realtime_analyzing = False
        self.state.realtime_fps = 0.0
        self.sig_status.emit("就绪")
        self.sig_log.emit("[实时] 已停止实时模式")
        self._emit_state()

    def start_realtime_analysis(self) -> None:
        if not self.state.realtime_running:
            self.sig_error.emit("提示", "请先启动实时模式")
            return
        if self.state.realtime_analyzing:
            return
        self._stop_realtime_thread()
        self.state.realtime_analyzing = True
        self.sig_status.emit("实时分析中…")
        self.sig_log.emit("[实时] 已开始实时分析")
        analysis_fps = float(self._config.get("realtime", {}).get("analysis_fps", 3.0))
        self._rt_thread = _RealtimeThread(self._rt_pipeline, analyze=True, target_fps=analysis_fps)
        self._rt_thread.sig_result.connect(self._on_realtime_result, Qt.QueuedConnection)
        self._rt_thread.sig_error.connect(self._on_realtime_error, Qt.QueuedConnection)
        self._rt_thread.start()
        self._emit_state()

    def stop_realtime_analysis(self) -> None:
        if not self.state.realtime_running:
            return
        self._stop_realtime_thread()
        self.state.realtime_analyzing = False
        # 若 pipeline 已被 stop_realtime_mode 清空（极端并发情况），直接返回
        if self._rt_pipeline is None:
            self._emit_state()
            return
        self.sig_status.emit("实时模式已启动")
        self.sig_log.emit("[实时] 已停止实时分析")
        raw_fps = float(self._config.get("realtime", {}).get("raw_fps", 5.0))
        self._rt_thread = _RealtimeThread(self._rt_pipeline, analyze=False, target_fps=raw_fps)
        self._rt_thread.sig_result.connect(self._on_realtime_result, Qt.QueuedConnection)
        self._rt_thread.sig_error.connect(self._on_realtime_error, Qt.QueuedConnection)
        self._rt_thread.start()
        self._emit_state()

    def _on_realtime_result(self, rt_res, fps: float) -> None:
        frame = rt_res.frame
        xyz = np.asarray(frame.points_xyz, dtype=np.float32)
        self.state.realtime_fps = float(fps)
        self.state.realtime_points = int(xyz.shape[0])
        self.state.current_file = frame.source_path
        self.state.loaded_pcd = numpy_xyz_to_pointcloud(xyz)

        if rt_res.scene is not None:
            # 只更新 last_scene，不拆散 last_det/last_seg；
            # 离线"融合显示"走 last_scene 分支，不会因类型不匹配而出错
            self.state.last_scene = rt_res.scene
            # 通知 main_window 刷新实时 Open3D 窗口
            self.sig_realtime_frame.emit(rt_res.scene)

        self._emit_state()

    def _on_realtime_error(self, title: str, msg: str) -> None:
        self.sig_error.emit(title, msg)
        self.stop_realtime_mode()
        # 线程崩溃后自动切回离线模式，避免用户被"困"在实时模式界面
        self.state.runtime_mode = "offline"
        self.sig_log.emit("[实时] 模式已自动切回离线，请检查错误原因后重试")
        self._emit_state()

    def clear_results(self) -> None:
        self.state.last_det = None
        self.state.last_seg = None
        self.state.last_scene = None
        self.sig_log.emit("已清空结果")
        self.sig_status.emit("就绪")
        self._emit_state()

