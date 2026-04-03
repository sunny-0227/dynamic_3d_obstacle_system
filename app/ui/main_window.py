"""
主窗口 — 毕业设计答辩展示版 UI

布局：顶栏标题区 | 左控制区（固定宽） + 右摘要与日志 | 底部分区状态栏。
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
from app.ui.defense_file_dialogs import pick_nuscenes_root_directory, pick_pointcloud_file
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

# 课题副标题（可通过 config app.thesis_title 覆盖）
DEFAULT_THESIS_TITLE = "面向动态场景的三维障碍物检测与分割系统研究"


class MainWindow(QMainWindow):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self._config = config or {}
        self._project_root = Path(__file__).resolve().parent.parent.parent

        self._controller = AppController(config=self._config)
        self._renderer: Optional[SceneRenderer] = None

        self._build_ui()
        self._apply_style()
        self._wire()

        self._controller.sig_log.emit("主窗口初始化完成（答辩展示版 UI）")
        QTimer.singleShot(0, lambda: self._on_state(self._controller.state))

    @staticmethod
    def _defer(fn: Callable[..., None], *args, **kwargs) -> None:
        QTimer.singleShot(0, partial(fn, *args, **kwargs))

    def _build_ui(self) -> None:
        app_cfg = self._config.get("app", {})
        app_name = app_cfg.get("name", "动态3D障碍物感知系统")
        thesis = app_cfg.get("thesis_title", DEFAULT_THESIS_TITLE)

        self.setWindowTitle(f"{app_name} ｜ 答辩演示")
        self.setMinimumSize(1120, 780)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(12, 12, 12, 8)
        outer.setSpacing(10)

        # 顶部标题区
        self._header = DefenseHeader(app_name, thesis, self)
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
        self._log = LogPanel(max_lines=800)
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

    def _on_controller_status(self, message: str) -> None:
        msg = str(message)
        self._header.set_status_line(f"当前状态：{msg}")
        self._dstatus.set_exec_status(f"执行：{msg}")

    def _on_panel_data_source(self, src: str) -> None:
        """切换数据源单选：必要时确认并清理另一侧状态。"""
        st = self._controller.state
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
            self._log.append("融合显示：使用一键运行生成的融合场景（scene）")
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

        sync_control_panel_to_state(self._panel, state)

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
                self._log.append("显示：融合场景（来自一键运行）")
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
