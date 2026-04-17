"""
主窗口 — 三维障碍物检测与分割科研系统 UI

布局：顶栏标题区 | 左控制区（固定宽） + 右摘要与系统日志 | 底部分区状态栏。
保持与 AppController 的既有信号/槽约定，不修改算法与数据集逻辑。
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.core.fusion.result_fusion import fuse_partial_for_gui_display
from app.ui.controller import AppController, AppState
from app.ui.defense_dialogs import (
    ask_nusc_root_clears_single_file,
    ask_pick_file_disconnects_nusc,
    ask_switch_data_source_to_nuscenes_clears_single,
    ask_switch_data_source_to_single_disconnects_nusc,
    info_need_nusc_data_source_for_connect,
    info_need_nusc_data_source_for_root,
    info_need_single_data_source_for_load,
    info_need_single_data_source_for_pick_file,
    show_error_critical,
    warn_empty_pcd_cannot_preview,
    warn_need_nonempty_pcd_for_fusion,
)
from app.ui.defense_file_dialogs import (
    pick_nuscenes_root_directory,
    pick_pointcloud_file,
    pick_realtime_stream_directory,
)
from app.ui.defense_panel_sync import (
    compute_action_button_flags,
    refresh_nusc_meta_line,
    sync_control_panel_to_state,
)
from app.ui.defense_presenter import (
    apply_defense_status_bar,
    build_summary_text,
    mode_header_text,
)
from app.ui.defense_styles import DEFENSE_MAINWINDOW_STYLESHEET
from app.ui.widgets.control_panel import ControlPanel
from app.ui.widgets.data_summary_card import DataSummaryCard
from app.ui.widgets.defense_header import DefenseHeader
from app.ui.widgets.defense_status_bar import DefenseStatusBar
from app.ui.widgets.log_panel import LogPanel
from app.utils.logger import get_logger
from app.visualization.open3d_viewer import show_pointcloud
from app.visualization.scene_renderer import RenderOptions, SceneRenderer

logger = get_logger("ui.main_window_defense")

# 主标题、副标题默认值（可通过 config app.main_title / app.subtitle 覆盖；兼容旧键 app.thesis_title）
DEFAULT_MAIN_TITLE = "面向动态场景的三维障碍物检测与分割系统研究"
DEFAULT_SUBTITLE = "三维点云感知与分析系统"


class MainWindow(QMainWindow):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self._config = config or {}
        self._project_root = Path(__file__).resolve().parent.parent.parent

        self._controller = AppController(config=self._config)
        self._renderer: Optional[SceneRenderer] = None          # 离线阻塞式渲染器
        self._rt_renderer: Optional[SceneRenderer] = None       # 实时非阻塞渲染器
        self._rt_tick_timer: Optional[QTimer] = None            # 驱动 Open3D 事件循环的定时器

        self._build_ui()
        self._apply_style()
        self._wire()

        self._controller.sig_log.emit("主窗口初始化完成")
        QTimer.singleShot(0, lambda: self._on_state(self._controller.state))

    @staticmethod
    def _defer(fn: Callable[..., None], *args, **kwargs) -> None:
        QTimer.singleShot(0, partial(fn, *args, **kwargs))

    def _build_ui(self) -> None:
        app_cfg = self._config.get("app", {})
        # 主标题：优先 main_title，其次兼容旧配置 thesis_title
        main_title = (
            app_cfg.get("main_title")
            or app_cfg.get("thesis_title")
            or DEFAULT_MAIN_TITLE
        )
        subtitle = app_cfg.get("subtitle") or DEFAULT_SUBTITLE

        self.setWindowTitle(main_title)
        self.setMinimumSize(1120, 780)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 8)
        outer.setSpacing(10)

        # 顶部标题区
        self._header = DefenseHeader(main_title, subtitle, self)
        outer.addWidget(self._header)

        # 左右分栏
        split = QHBoxLayout()
        split.setSpacing(14)

        self._panel = ControlPanel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(self._panel)
        split.addWidget(scroll, stretch=0)

        right = QVBoxLayout()
        right.setSpacing(10)
        self._summary = DataSummaryCard()
        self._log = LogPanel(title="系统日志", max_lines=800)
        right.addWidget(self._summary, stretch=0)
        right.addWidget(self._log, stretch=1)
        split.addLayout(right, stretch=1)

        outer.addLayout(split, stretch=1)

        # 底部状态栏（三分栏）
        self._dstatus = DefenseStatusBar()
        self.setStatusBar(self._dstatus)

    def _apply_style(self) -> None:
        self.setStyleSheet(DEFENSE_MAINWINDOW_STYLESHEET)

    def _wire(self) -> None:
        self._panel.sig_runtime_mode_changed.connect(self._on_runtime_mode_changed)
        self._panel.sig_data_source_changed.connect(self._on_panel_data_source)

        self._panel.sig_select_file.connect(self._on_pick_file)
        self._panel.sig_load_pointcloud.connect(self._guard_load_single_file)

        self._panel.sig_select_nusc_root.connect(self._on_pick_nusc_root)
        self._panel.sig_connect_nusc.connect(self._guard_connect_nusc)
        self._panel.sig_nav_changed.connect(self._on_nav_changed)
        self._panel.sig_scene_changed.connect(self._on_scene_changed)
        self._panel.sig_prev_frame.connect(self._on_prev_frame)
        self._panel.sig_next_frame.connect(self._on_next_frame)
        self._panel.sig_load_current_frame.connect(
            lambda: self._controller.load_nusc_frame(self._panel.frame_index())
        )
        self._panel.sig_select_realtime_dir.connect(self._on_pick_realtime_dir)
        self._panel.sig_start_realtime.connect(self._controller.start_realtime_mode)
        self._panel.sig_stop_realtime.connect(self._controller.stop_realtime_mode)
        self._panel.sig_start_realtime_analysis.connect(self._controller.start_realtime_analysis)
        self._panel.sig_stop_realtime_analysis.connect(self._controller.stop_realtime_analysis)

        self._panel.sig_run_detect.connect(self._controller.run_detect)
        self._panel.sig_run_segment.connect(self._controller.run_segment)
        self._panel.sig_run_full.connect(self._controller.run_full)
        self._panel.sig_clear_results.connect(self._on_clear)
        self._panel.sig_show_fusion.connect(self._on_show_fusion)

        self._controller.sig_log.connect(self._log.append)
        self._controller.sig_status.connect(self._on_controller_status)
        self._controller.sig_error.connect(
            lambda t, m: show_error_critical(self, t, m)
        )
        self._controller.sig_state.connect(self._on_state)
        self._controller.sig_busy.connect(self._panel.set_busy)
        self._controller.sig_request_render.connect(self._on_render_request)
        self._controller.sig_realtime_frame.connect(self._on_realtime_frame)

    def _on_runtime_mode_changed(self, mode: str) -> None:
        if mode == "offline" and self._controller.state.realtime_running:
            self._controller.stop_realtime_mode()
        if mode == "offline":
            # 切回离线模式时关闭实时渲染窗口
            self._close_realtime_renderer()
        self._controller.set_runtime_mode("realtime" if mode == "realtime" else "offline")
        self._panel.apply_runtime_module_lock(mode)
        self._on_state(self._controller.state)

    def _on_controller_status(self, message: str) -> None:
        msg = str(message)
        self._header.set_status_line(f"系统状态：{msg}")
        self._dstatus.set_exec_status(f"任务状态：{msg}")

    def _on_panel_data_source(self, src: str) -> None:
        """切换数据源单选：必要时确认并清理另一侧状态。"""
        st = self._controller.state
        if st.runtime_mode != "offline":
            return
        if src == "nuscenes":
            if st.workflow == "single_file" and st.current_file is not None:
                if not ask_switch_data_source_to_nuscenes_clears_single(self):
                    self._panel.sync_data_source_radio("single_file")
                    return
                self._controller.set_current_file(None)
                self._panel.set_selected_file(None)
        else:
            if st.workflow == "nuscenes" and st.nusc_loader is not None and st.nusc_loader.is_connected:
                if not ask_switch_data_source_to_single_disconnects_nusc(self):
                    self._panel.sync_data_source_radio("nuscenes")
                    return
                self._controller.disconnect_nusc()
                self._panel.set_nusc_root(None)

        self._panel.apply_source_module_lock(src)
        self._on_state(self._controller.state)

    def _on_pick_realtime_dir(self) -> None:
        root = pick_realtime_stream_directory(self, self._project_root)
        if root is None:
            self._log.append("已取消选择实时流目录")
            return
        self._panel.set_realtime_stream_dir(root)
        self._controller.set_realtime_stream_dir(root)

    def _guard_load_single_file(self) -> None:
        if self._panel.ui_data_source() != "single_file":
            info_need_single_data_source_for_load(self)
            return
        self._controller.load_current_file_pointcloud()

    def _guard_connect_nusc(self) -> None:
        if self._panel.ui_data_source() != "nuscenes":
            info_need_nusc_data_source_for_connect(self)
            return
        self._on_connect_nusc_clicked()

    def _on_pick_file(self) -> None:
        if self._panel.ui_data_source() != "single_file":
            info_need_single_data_source_for_pick_file(self)
            return

        st = self._controller.state
        if st.workflow == "nuscenes" and st.nusc_loader is not None and st.nusc_loader.is_connected:
            if not ask_pick_file_disconnects_nusc(self):
                self._log.append("已取消：保留 nuScenes 模式")
                return
            self._controller.disconnect_nusc()
            self._panel.set_nusc_root(None)
            self._panel.sync_data_source_radio("single_file")
            self._panel.apply_source_module_lock("single_file")

        pc_cfg = self._config.get("pointcloud", {})
        p = pick_pointcloud_file(self, self._project_root, pc_cfg)
        if p is None:
            self._log.append("已取消选择文件")
            return
        self._panel.set_selected_file(p)
        self._controller.set_current_file(p)

    def _on_pick_nusc_root(self) -> None:
        if self._panel.ui_data_source() != "nuscenes":
            info_need_nusc_data_source_for_root(self)
            return

        st = self._controller.state
        if st.workflow == "single_file" and st.current_file is not None:
            if not ask_nusc_root_clears_single_file(self):
                self._log.append("已取消：保留单文件模式")
                return
            self._controller.set_current_file(None)
            self._panel.set_selected_file(None)

        root = pick_nuscenes_root_directory(self, self._project_root)
        if root is None:
            self._log.append("已取消选择 nuScenes 根目录")
            return
        self._panel.set_nusc_root(root)
        self._controller.set_nusc_root(root)

    def _on_connect_nusc_clicked(self) -> None:
        self._controller.connect_nusc(
            nav_mode=self._panel.navigation_mode(),
            scene_token=self._panel.current_scene_token(),
        )

    def _on_nav_changed(self, mode: str) -> None:
        token = self._panel.current_scene_token()
        self._controller.set_nusc_navigation(mode, scene_token=token)
        self._update_nusc_meta()

    def _on_scene_changed(self) -> None:
        if self._panel.navigation_mode() != "scene":
            return
        self._controller.set_nusc_navigation(
            "scene", scene_token=self._panel.current_scene_token()
        )
        self._update_nusc_meta()

    def _on_prev_frame(self) -> None:
        idx = max(0, self._panel.frame_index() - 1)
        self._panel.set_frame_index(idx)
        self._update_nusc_meta()

    def _on_next_frame(self) -> None:
        idx = self._panel.frame_index() + 1
        self._panel.set_frame_index(idx)
        self._update_nusc_meta()

    def _on_clear(self) -> None:
        self._controller.clear_results()
        self._log.clear()
        self._log.append("日志已清空，结果已清空")

    def _on_show_fusion(self) -> None:
        st: AppState = self._controller.state
        if st.loaded_pcd is None or len(st.loaded_pcd.points) == 0:
            warn_need_nonempty_pcd_for_fusion(self)
            return

        if st.last_scene is not None:
            self._log.append("融合显示：使用一键分析生成的融合场景（scene）")
            self._render_scene(st.last_scene)
            return

        pts_xyz = np.asarray(st.loaded_pcd.points, dtype=np.float32)

        try:
            scene, primary, extra_lines = fuse_partial_for_gui_display(
                pts_xyz, st.last_seg, st.last_det
            )
            self._log.append(primary)
            for line in extra_lines:
                self._log.append(line)
            self._render_scene(scene)
            return
        except Exception as e:
            logger.warning("融合显示组装失败，将回退为简单显示：%s", e)

        if st.last_seg is not None and getattr(st.last_seg, "colored_pcd", None) is not None:
            self._log.append("融合显示：回退为仅分割点云显示")
            show_pointcloud(st.last_seg.colored_pcd, window_title="融合显示（仅分割点云）")
        else:
            self._log.append("融合显示：回退为仅原始点云显示")
            show_pointcloud(st.loaded_pcd, window_title="融合显示（仅原始点云）")

    def _on_state(self, state: AppState) -> None:
        loader = state.nusc_loader
        nusc_connected = loader is not None and loader.is_connected

        self._panel.sync_runtime_mode_radio(state.runtime_mode)
        self._panel.apply_runtime_module_lock(state.runtime_mode)
        sync_control_panel_to_state(self._panel, state)
        self._panel.set_realtime_stream_dir(state.realtime_stream_dir)
        self._panel.set_realtime_controls(
            running=bool(state.realtime_running),
            analyzing=bool(state.realtime_analyzing),
        )
        self._panel.set_realtime_stats(
            fps=float(state.realtime_fps),
            points=int(state.realtime_points),
            source=str(state.realtime_source),
        )

        has_nonempty, has_results, allow_autoload = compute_action_button_flags(state)
        self._panel.set_action_buttons_state(
            has_nonempty_pcd=has_nonempty,
            has_results=has_results,
            allow_run_full_autoload=allow_autoload,
            defense_strict_pipeline=True,
        )

        self._header.set_mode_line(mode_header_text(self._panel, state, nusc_connected))
        self._summary.set_summary(
            build_summary_text(self._panel, state, loader, nusc_connected)
        )
        apply_defense_status_bar(self._dstatus, state, loader, nusc_connected)

    def _update_nusc_meta(self) -> None:
        loader = self._controller.state.nusc_loader
        if loader is None or not loader.is_connected:
            return
        refresh_nusc_meta_line(self._panel, loader)

    def _on_render_request(self, mode: str) -> None:
        st: AppState = self._controller.state
        if mode == "seg":
            if st.last_seg is not None and getattr(st.last_seg, "colored_pcd", None) is not None:
                self._log.append("显示：语义分割结果（彩色点云）")
                self._defer(
                    show_pointcloud,
                    st.last_seg.colored_pcd,
                    window_title="语义分割结果（彩色）",
                )
            elif st.last_seg is not None:
                # 分割成功但着色/Open3D 点云未生成时，避免「自动预览」静默无反馈
                self._log.append(
                    "显示：分割已完成，但未生成彩色点云，已跳过 Open3D 预览。"
                )
                QMessageBox.information(
                    self,
                    "提示",
                    "分割已完成，但当前结果未生成彩色点云，无法打开 Open3D 彩色预览。\n"
                    "可使用「融合显示」或右侧摘要查看分割相关信息。",
                )
            return
        if mode == "raw":
            if st.loaded_pcd is not None and len(st.loaded_pcd.points) > 0:
                self._log.append("显示：原始点云预览")
                self._defer(show_pointcloud, st.loaded_pcd, window_title="点云预览")
            elif st.loaded_pcd is not None:
                warn_empty_pcd_cannot_preview(self)
            return
        if mode == "fusion":
            if st.last_scene is not None:
                self._log.append("显示：融合场景（来自一键分析）")
                self._defer(self._render_scene, st.last_scene)
            else:
                self._defer(self._on_show_fusion)

    def _render_scene(self, scene) -> None:
        if self._renderer is None:
            vis_cfg = self._config.get("visualization", {})
            self._renderer = SceneRenderer(
                RenderOptions(
                    window_title=vis_cfg.get("window_title", "融合显示（点云+分割+检测框）"),
                    width=int(vis_cfg.get("window_width", 1280)),
                    height=int(vis_cfg.get("window_height", 720)),
                    background_color=vis_cfg.get("background_color", [0.05, 0.05, 0.05]),
                    point_size=float(vis_cfg.get("point_size", 2.0)),
                    show_coordinate_frame=True,
                )
            )

        self._panel.set_busy(True)
        try:
            self._renderer.render(scene)
        finally:
            self._panel.set_busy(False)

    # ------------------------------------------------------------------
    # 实时 Open3D 窗口管理
    # ------------------------------------------------------------------

    def _ensure_realtime_renderer(self) -> SceneRenderer:
        """懒创建实时渲染器并打开 Open3D 窗口（仅第一次调用时创建）。"""
        if self._rt_renderer is None or not self._rt_renderer.is_open:
            vis_cfg = self._config.get("visualization", {})
            self._rt_renderer = SceneRenderer(
                RenderOptions(
                    window_title="实时三维感知（点云 + 分割 + 检测框）",
                    width=int(vis_cfg.get("window_width", 1280)),
                    height=int(vis_cfg.get("window_height", 720)),
                    background_color=vis_cfg.get("background_color", [0.05, 0.05, 0.05]),
                    point_size=float(vis_cfg.get("point_size", 2.0)),
                    show_coordinate_frame=True,
                )
            )
            self._rt_renderer.open_realtime_window()

            # 启动定时器，持续驱动 Open3D 事件循环（33ms ≈ 30fps 的界面响应上限）
            if self._rt_tick_timer is None:
                self._rt_tick_timer = QTimer(self)
                self._rt_tick_timer.timeout.connect(self._on_rt_tick)
            self._rt_tick_timer.start(33)
            self._log.append("[实时] Open3D 可视化窗口已打开")

        return self._rt_renderer

    def _on_realtime_frame(self, scene) -> None:
        """
        接收控制器发来的实时帧 FusedScene，刷新 Open3D 窗口。

        由 controller.sig_realtime_frame 信号驱动，在 GUI 主线程中执行。
        """
        renderer = self._ensure_realtime_renderer()
        still_open = renderer.update(scene)
        if not still_open:
            # 用户手动关闭了 Open3D 窗口，停止实时分析
            self._log.append("[实时] Open3D 窗口已被关闭，停止实时分析")
            self._controller.stop_realtime_analysis()
            self._close_realtime_renderer()

    def _on_rt_tick(self) -> None:
        """
        定时器回调：在没有新帧时驱动 Open3D 事件循环，保持窗口响应鼠标/键盘。

        注意：update() 已包含 poll_events + update_renderer，
        tick() 只在两帧之间补驱动，避免窗口假死。
        """
        if self._rt_renderer is None or not self._rt_renderer.is_open:
            if self._rt_tick_timer is not None:
                self._rt_tick_timer.stop()
            return
        still_open = self._rt_renderer.tick()
        if not still_open:
            self._log.append("[实时] Open3D 窗口已被关闭（tick 检测）")
            self._controller.stop_realtime_analysis()
            self._close_realtime_renderer()

    def _close_realtime_renderer(self) -> None:
        """关闭实时渲染窗口并停止定时器（幂等）。"""
        if self._rt_tick_timer is not None:
            self._rt_tick_timer.stop()
        if self._rt_renderer is not None:
            self._rt_renderer.close()
            self._rt_renderer = None
