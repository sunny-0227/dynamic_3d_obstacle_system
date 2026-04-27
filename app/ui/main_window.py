"""
主窗口 — 面向动态场景的三维障碍物检测与分割系统研究

布局：左侧导航栏（固定宽 180px）+ 右侧 QStackedWidget（4 页）。
  Page 0 — 离线点云分析（OfflinePage）
  Page 1 — 实时相机感知（RealtimePage）
  Page 2 — 模型与配置（ConfigPage）
  Page 3 — 系统日志（LogPage）

所有核心业务逻辑（Controller、Open3D、OpenPCDet、RealSense）保持不变，
仅替换 UI 层的布局与控件组织方式。
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
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
from app.ui.defense_panel_sync import compute_action_button_flags, refresh_nusc_meta_line
from app.ui.defense_styles import DEFENSE_MAINWINDOW_STYLESHEET
from app.ui.pages.config_page import ConfigPage
from app.ui.pages.log_page import LogPage
from app.ui.pages.offline_page import OfflinePage
from app.ui.pages.realtime_page import RealtimePage
from app.ui.widgets.defense_status_bar import DefenseStatusBar
from app.utils.logger import get_logger
from app.visualization.open3d_viewer import show_pointcloud
from app.visualization.scene_renderer import RenderOptions, SceneRenderer

logger = get_logger("ui.main_window")

DEFAULT_MAIN_TITLE = "面向动态场景的三维障碍物检测与分割系统研究"
DEFAULT_SUBTITLE   = "三维点云感知与分析系统"

# 导航项 (文字标签, 图标文字, 对应 stacked index)
_NAV_ITEMS: List[tuple[str, str, int]] = [
    ("离线点云分析",  "📂", 0),
    ("实时相机感知",  "📷", 1),
    ("模型与配置",    "⚙",  2),
    ("系统日志",      "📋", 3),
]


class MainWindow(QMainWindow):
    def __init__(self, config: dict | None = None):
        super().__init__()
        self._config       = config or {}
        self._project_root = Path(__file__).resolve().parent.parent.parent

        self._controller: AppController = AppController(config=self._config)
        self._renderer:     Optional[SceneRenderer] = None   # 离线阻塞式渲染器
        self._rt_renderer:  Optional[SceneRenderer] = None   # 实时非阻塞渲染器
        self._rt_tick_timer: Optional[QTimer]       = None   # 驱动 Open3D 事件循环

        self._nav_buttons: List[QPushButton] = []

        self._build_ui()
        self._apply_style()
        self._wire()

        self._controller.sig_log.emit("系统初始化完成")
        QTimer.singleShot(0, lambda: self._on_state(self._controller.state))

    # ------------------------------------------------------------------
    # 构建 UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        app_cfg    = self._config.get("app", {})
        main_title = app_cfg.get("main_title") or app_cfg.get("thesis_title") or DEFAULT_MAIN_TITLE
        subtitle   = app_cfg.get("subtitle") or DEFAULT_SUBTITLE

        self.setWindowTitle(main_title)
        self.setMinimumSize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── 顶部标题栏 ───────────────────────────────────────────
        self._title_bar = self._build_title_bar(main_title, subtitle)
        outer.addWidget(self._title_bar)

        # ── 主体区（导航 + 内容） ─────────────────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_nav_bar(), stretch=0)
        body.addWidget(self._build_content_area(), stretch=1)
        outer.addLayout(body, stretch=1)

        # ── 底部状态栏 ────────────────────────────────────────────
        self._dstatus = DefenseStatusBar()
        self.setStatusBar(self._dstatus)

    def _build_title_bar(self, main_title: str, subtitle: str) -> QWidget:
        bar = QWidget()
        bar.setObjectName("titleBar")
        bar.setFixedHeight(72)
        bar.setStyleSheet(
            "QWidget#titleBar {"
            "  background-color: #181825;"
            "  border-bottom: 1px solid #313244;"
            "}"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(24, 8, 24, 8)
        lay.setSpacing(0)

        left = QVBoxLayout()
        left.setSpacing(2)
        self._lbl_main_title = QLabel(main_title)
        self._lbl_main_title.setObjectName("mainTitle")
        self._lbl_subtitle = QLabel(subtitle)
        self._lbl_subtitle.setObjectName("subTitle")
        left.addWidget(self._lbl_main_title)
        left.addWidget(self._lbl_subtitle)
        lay.addLayout(left, stretch=1)

        # 右侧状态文字
        self._lbl_status = QLabel("系统就绪")
        self._lbl_status.setObjectName("statusLine")
        self._lbl_status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay.addWidget(self._lbl_status)

        return bar

    def _build_nav_bar(self) -> QWidget:
        nav = QWidget()
        nav.setObjectName("navBar")
        nav.setFixedWidth(180)

        lay = QVBoxLayout(nav)
        lay.setContentsMargins(0, 16, 0, 16)
        lay.setSpacing(0)

        # 系统简称
        lbl_sys = QLabel("导航")
        lbl_sys.setObjectName("sysTitle")
        lbl_sys.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lay.addWidget(lbl_sys)
        lay.addSpacerItem(QSpacerItem(0, 12, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # 导航按钮
        for label, icon, idx in _NAV_ITEMS:
            btn = QPushButton(f" {icon}  {label}")
            btn.setObjectName("navBtn")
            btn.setFixedHeight(48)
            btn.setProperty("active", "false")
            btn.setCheckable(False)
            btn.clicked.connect(partial(self._switch_page, idx))
            self._nav_buttons.append(btn)
            lay.addWidget(btn)

        lay.addStretch(1)

        # 版本号
        ver = self._config.get("app", {}).get("version", "1.0.0")
        lbl_ver = QLabel(f"v{ver}")
        lbl_ver.setObjectName("sysSubtitle")
        lbl_ver.setAlignment(Qt.AlignCenter)
        lay.addWidget(lbl_ver)

        return nav

    def _build_content_area(self) -> QStackedWidget:
        self._stack = QStackedWidget()

        # Page 0: 离线点云分析
        self._offline_page = OfflinePage()
        self._stack.addWidget(self._offline_page)

        # Page 1: 实时相机感知
        self._realtime_page = RealtimePage()
        self._stack.addWidget(self._realtime_page)

        # Page 2: 模型与配置
        self._config_page = ConfigPage(self._config, self._project_root)
        self._stack.addWidget(self._config_page)

        # Page 3: 系统日志
        self._log_page = LogPage()
        self._log_page.set_project_root(self._project_root)
        self._stack.addWidget(self._log_page)

        self._stack.setCurrentIndex(0)
        return self._stack

    def _apply_style(self) -> None:
        self.setStyleSheet(DEFENSE_MAINWINDOW_STYLESHEET)
        # 初始激活第一个导航按钮
        self._set_nav_active(0)

    # ------------------------------------------------------------------
    # 导航切换
    # ------------------------------------------------------------------

    def _switch_page(self, idx: int) -> None:
        """切换右侧内容页并更新导航按钮激活状态。"""
        self._stack.setCurrentIndex(idx)
        self._set_nav_active(idx)

    def _set_nav_active(self, active_idx: int) -> None:
        for i, btn in enumerate(self._nav_buttons):
            is_active = i == active_idx
            btn.setProperty("active", "true" if is_active else "false")
            # 刷新样式（QSS 属性变更必须 unpolish/polish）
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ------------------------------------------------------------------
    # 信号连接
    # ------------------------------------------------------------------

    def _wire(self) -> None:
        self._wire_offline()
        self._wire_realtime()
        self._wire_controller()

    def _wire_offline(self) -> None:
        p = self._offline_page
        p.sig_data_source_changed.connect(self._on_panel_data_source)
        p.sig_select_file.connect(self._on_pick_file)
        p.sig_load_pointcloud.connect(self._guard_load_single_file)
        p.sig_select_nusc_root.connect(self._on_pick_nusc_root)
        p.sig_connect_nusc.connect(self._guard_connect_nusc)
        p.sig_nav_changed.connect(self._on_nav_changed)
        p.sig_scene_changed.connect(self._on_scene_changed)
        p.sig_prev_frame.connect(self._on_prev_frame)
        p.sig_next_frame.connect(self._on_next_frame)
        p.sig_load_current_frame.connect(
            lambda: self._controller.load_nusc_frame(self._offline_page.frame_index())
        )
        p.sig_run_detect.connect(self._controller.run_detect)
        p.sig_run_segment.connect(self._controller.run_segment)
        p.sig_show_fusion.connect(self._on_show_fusion)
        p.sig_run_full.connect(self._controller.run_full)
        p.sig_clear_results.connect(self._on_clear)

    def _wire_realtime(self) -> None:
        p = self._realtime_page
        p.sig_camera_source_changed.connect(self._on_camera_source_changed)
        p.sig_select_realtime_dir.connect(self._on_pick_realtime_dir)
        p.sig_start_realtime.connect(self._on_start_realtime)
        p.sig_stop_realtime.connect(self._on_stop_realtime)
        p.sig_start_realtime_analysis.connect(self._controller.start_realtime_analysis)
        p.sig_stop_realtime_analysis.connect(self._controller.stop_realtime_analysis)

    def _wire_controller(self) -> None:
        c = self._controller
        c.sig_log.connect(self._log_page.append)
        c.sig_status.connect(self._on_controller_status)
        c.sig_error.connect(lambda t, m: show_error_critical(self, t, m))
        c.sig_state.connect(self._on_state)
        c.sig_busy.connect(self._on_busy)
        c.sig_request_render.connect(self._on_render_request)
        c.sig_realtime_frame.connect(self._on_realtime_frame)
        # 配置保存成功后写入日志，提示用户重启
        self._config_page.sig_config_saved.connect(
            lambda: self._log_page.append("[配置] 配置已保存到 settings.yaml，请重启应用以使新参数生效")
        )

    # ------------------------------------------------------------------
    # 离线模式事件处理
    # ------------------------------------------------------------------

    def _on_panel_data_source(self, src: str) -> None:
        st = self._controller.state
        if st.runtime_mode != "offline":
            return

        if src == "nuscenes":
            if st.workflow == "single_file" and st.current_file is not None:
                if not ask_switch_data_source_to_nuscenes_clears_single(self):
                    self._offline_page.sync_data_source_radio("single_file")
                    return
                self._controller.set_current_file(None)
                self._offline_page.set_selected_file(None)
        else:
            if st.workflow == "nuscenes" and st.nusc_loader is not None and st.nusc_loader.is_connected:
                if not ask_switch_data_source_to_single_disconnects_nusc(self):
                    self._offline_page.sync_data_source_radio("nuscenes")
                    return
                self._controller.disconnect_nusc()
                self._offline_page.set_nusc_root(None)

        self._offline_page.sync_data_source_radio(src)
        self._on_state(self._controller.state)

    def _guard_load_single_file(self) -> None:
        if self._offline_page.ui_data_source() != "single_file":
            info_need_single_data_source_for_load(self)
            return
        self._controller.load_current_file_pointcloud()

    def _guard_connect_nusc(self) -> None:
        if self._offline_page.ui_data_source() != "nuscenes":
            info_need_nusc_data_source_for_connect(self)
            return
        self._controller.connect_nusc(
            nav_mode=self._offline_page.navigation_mode(),
            scene_token=self._offline_page.current_scene_token(),
        )

    def _on_pick_file(self) -> None:
        if self._offline_page.ui_data_source() != "single_file":
            info_need_single_data_source_for_pick_file(self)
            return

        st = self._controller.state
        if st.workflow == "nuscenes" and st.nusc_loader is not None and st.nusc_loader.is_connected:
            if not ask_pick_file_disconnects_nusc(self):
                return
            self._controller.disconnect_nusc()
            self._offline_page.set_nusc_root(None)
            self._offline_page.sync_data_source_radio("single_file")

        pc_cfg = self._config.get("pointcloud", {})
        p = pick_pointcloud_file(self, self._project_root, pc_cfg)
        if p is None:
            return
        self._offline_page.set_selected_file(p)
        self._controller.set_current_file(p)

    def _on_pick_nusc_root(self) -> None:
        if self._offline_page.ui_data_source() != "nuscenes":
            info_need_nusc_data_source_for_root(self)
            return

        st = self._controller.state
        if st.workflow == "single_file" and st.current_file is not None:
            if not ask_nusc_root_clears_single_file(self):
                return
            self._controller.set_current_file(None)
            self._offline_page.set_selected_file(None)

        root = pick_nuscenes_root_directory(self, self._project_root)
        if root is None:
            return
        self._offline_page.set_nusc_root(root)
        self._controller.set_nusc_root(root)

    def _on_nav_changed(self, mode: str) -> None:
        token = self._offline_page.current_scene_token()
        self._controller.set_nusc_navigation(mode, scene_token=token)
        self._update_nusc_meta()

    def _on_scene_changed(self) -> None:
        if self._offline_page.navigation_mode() != "scene":
            return
        self._controller.set_nusc_navigation(
            "scene", scene_token=self._offline_page.current_scene_token()
        )
        self._update_nusc_meta()

    def _on_prev_frame(self) -> None:
        idx = max(0, self._offline_page.frame_index() - 1)
        self._offline_page.set_frame_index(idx)
        self._update_nusc_meta()

    def _on_next_frame(self) -> None:
        loader = self._controller.state.nusc_loader
        max_idx = (loader.frame_count - 1) if (loader is not None and loader.is_connected) else 0
        idx = min(self._offline_page.frame_index() + 1, max(0, max_idx))
        self._offline_page.set_frame_index(idx)
        self._update_nusc_meta()

    def _update_nusc_meta(self) -> None:
        loader = self._controller.state.nusc_loader
        if loader is None or not loader.is_connected:
            return
        refresh_nusc_meta_line(self._offline_page, loader)

    def _on_clear(self) -> None:
        self._controller.clear_results()
        self._log_page.clear()
        self._log_page.append("日志已清空，结果已重置")

    def _on_show_fusion(self) -> None:
        st: AppState = self._controller.state
        if st.loaded_pcd is None or len(st.loaded_pcd.points) == 0:
            warn_need_nonempty_pcd_for_fusion(self)
            return

        if st.last_scene is not None:
            self._log_page.append("融合显示：使用一键分析生成的融合场景")
            self._render_scene(st.last_scene)
            return

        pts_xyz = np.asarray(st.loaded_pcd.points, dtype=np.float32)
        try:
            scene, primary, extra_lines = fuse_partial_for_gui_display(
                pts_xyz, st.last_seg, st.last_det
            )
            self._log_page.append(primary)
            for line in extra_lines:
                self._log_page.append(line)
            self._render_scene(scene)
            return
        except Exception as e:
            logger.warning("融合显示组装失败，将回退为简单显示：%s", e)

        if st.last_seg is not None and getattr(st.last_seg, "colored_pcd", None) is not None:
            self._log_page.append("融合显示：回退为仅分割点云显示")
            show_pointcloud(st.last_seg.colored_pcd, window_title="融合显示（仅分割点云）")
        else:
            self._log_page.append("融合显示：回退为仅原始点云显示")
            show_pointcloud(st.loaded_pcd, window_title="融合显示（仅原始点云）")

    # ------------------------------------------------------------------
    # 实时模式事件处理
    # ------------------------------------------------------------------

    def _on_camera_source_changed(self, src: str) -> None:
        """
        界面切换数据源（realsense / mock）时，同步更新 config 中的 camera_type，
        使后续 start_realtime_mode 使用正确的相机实例。
        """
        if "realtime" not in self._config:
            self._config["realtime"] = {}
        self._config["realtime"]["camera_type"] = src
        self._log_page.append(f"数据源已切换为：{'RealSense' if src == 'realsense' else 'Mock'}")

    def _on_pick_realtime_dir(self) -> None:
        root = pick_realtime_stream_directory(self, self._project_root)
        if root is None:
            return
        self._realtime_page.set_realtime_stream_dir(root)
        self._controller.set_realtime_stream_dir(root)

    def _on_start_realtime(self) -> None:
        """启动实时模式前，确保 controller 的 config 使用界面选择的相机类型。"""
        src = self._realtime_page.camera_source()
        self._on_camera_source_changed(src)
        self._controller.start_realtime_mode()

    def _on_stop_realtime(self) -> None:
        """停止实时模式：先停止分析和渲染窗口，再通知 controller 停机。"""
        self._controller.stop_realtime_analysis()
        self._close_realtime_renderer()
        self._controller.stop_realtime_mode()

    # ------------------------------------------------------------------
    # Controller 信号处理
    # ------------------------------------------------------------------

    def _on_controller_status(self, message: str) -> None:
        self._lbl_status.setText(f"状态：{message}")
        self._dstatus.set_exec_status(f"任务状态：{message}")

    def _on_busy(self, busy: bool) -> None:
        self._offline_page.set_busy(busy)
        self._realtime_page.set_busy(busy)

    def _on_state(self, state: AppState) -> None:
        loader       = state.nusc_loader
        nusc_connected = loader is not None and loader.is_connected

        # ── 离线页面同步 ───────────────────────────────────────────
        op = self._offline_page

        # 数据源单选
        if state.workflow == "nuscenes":
            op.sync_data_source_radio("nuscenes")
        elif state.workflow == "single_file":
            op.sync_data_source_radio("single_file")

        op.set_selected_file(state.current_file)
        op.set_nusc_root(state.nusc_root)

        if nusc_connected:
            scenes = loader.get_scene_summaries()
            items  = [(s["name"], s["token"]) for s in scenes]
            op.set_scene_list(items)
            op.set_nusc_nav_enabled(True, frame_count=loader.frame_count)
            refresh_nusc_meta_line(op, loader)
        else:
            op.set_nusc_nav_enabled(False, frame_count=0)
            op.set_nusc_meta_text("请先选择根目录并加载数据集")

        # 操作按钮
        has_nonempty, has_results, allow_autoload = compute_action_button_flags(state)
        op.set_action_buttons_state(
            has_nonempty_pcd=has_nonempty,
            has_results=has_results,
            allow_run_full_autoload=allow_autoload,
            defense_strict_pipeline=True,
        )

        # 状态信息展示
        n_points = len(state.loaded_pcd.points) if state.loaded_pcd is not None else 0
        n_det    = len(state.last_det.boxes) if state.last_det is not None else -1
        n_seg    = int(state.last_seg.labels.max() + 1) if (
            state.last_seg is not None and hasattr(state.last_seg, "labels")
        ) else -1
        op.update_status(state.current_file, n_points, n_det, n_seg)

        # ── 实时页面同步 ──────────────────────────────────────────
        rp = self._realtime_page
        rp.set_controls(
            running=bool(state.realtime_running),
            analyzing=bool(state.realtime_analyzing),
        )
        rp.set_stats(
            fps=float(state.realtime_fps),
            points=int(state.realtime_points),
            obstacles=int(getattr(state, "realtime_obstacles", 0)),
            source=str(state.realtime_source),
            running=bool(state.realtime_running),
            analyzing=bool(state.realtime_analyzing),
        )

        # ── 底部状态栏 ─────────────────────────────────────────────
        self._sync_status_bar(state, loader, nusc_connected)

    def _sync_status_bar(self, state: AppState, loader, nusc_connected: bool) -> None:
        if nusc_connected and loader is not None:
            self._dstatus.set_dataset_status(
                f"数据集：已连接 ｜ {loader.mode_display_zh()} ｜ {loader.navigation_display_zh()}"
            )
        elif state.nusc_root:
            self._dstatus.set_dataset_status("数据集：已选根目录，待加载")
        else:
            self._dstatus.set_dataset_status("数据集：未选择")

        if state.loaded_pcd is not None and len(state.loaded_pcd.points) > 0:
            self._dstatus.set_frame_status(
                f"当前点云：已载入 ｜ {len(state.loaded_pcd.points):,} 点"
            )
        elif state.workflow == "nuscenes" and nusc_connected:
            self._dstatus.set_frame_status("当前帧：未载入 LiDAR 点云")
        else:
            self._dstatus.set_frame_status("当前帧：—")

    # ------------------------------------------------------------------
    # 渲染（离线）
    # ------------------------------------------------------------------

    def _on_render_request(self, mode: str) -> None:
        st: AppState = self._controller.state
        if mode == "seg":
            if st.last_seg is not None and getattr(st.last_seg, "colored_pcd", None) is not None:
                self._log_page.append("显示：语义分割结果（彩色点云）")
                QTimer.singleShot(
                    0, partial(show_pointcloud, st.last_seg.colored_pcd, "语义分割结果（彩色）")
                )
            elif st.last_seg is not None:
                self._log_page.append("显示：分割完成，但未生成彩色点云，已跳过 Open3D 预览")
                QMessageBox.information(
                    self, "提示",
                    "分割已完成，但当前结果未生成彩色点云，无法打开 Open3D 彩色预览。\n"
                    "可使用「融合显示」查看分割相关信息。",
                )
            return

        if mode == "raw":
            if st.loaded_pcd is not None and len(st.loaded_pcd.points) > 0:
                self._log_page.append("显示：原始点云预览")
                QTimer.singleShot(0, partial(show_pointcloud, st.loaded_pcd, "点云预览"))
            elif st.loaded_pcd is not None:
                warn_empty_pcd_cannot_preview(self)
            return

        if mode == "fusion":
            if st.last_scene is not None:
                self._log_page.append("显示：融合场景（来自一键分析）")
                QTimer.singleShot(0, partial(self._render_scene, st.last_scene))
            else:
                QTimer.singleShot(0, self._on_show_fusion)

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
        self._on_busy(True)
        try:
            self._renderer.render(scene)
        finally:
            self._on_busy(False)

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

            if self._rt_tick_timer is None:
                self._rt_tick_timer = QTimer(self)
                self._rt_tick_timer.timeout.connect(self._on_rt_tick)
            self._rt_tick_timer.start(33)
            self._log_page.append("[实时] Open3D 可视化窗口已打开")

        return self._rt_renderer

    def _on_realtime_frame(self, scene) -> None:
        """接收控制器发来的实时帧 FusedScene，刷新 Open3D 窗口。"""
        renderer   = self._ensure_realtime_renderer()
        still_open = renderer.update(scene)
        if not still_open:
            self._log_page.append("[实时] Open3D 窗口已被关闭，停止实时分析")
            self._controller.stop_realtime_analysis()
            self._close_realtime_renderer()

    def _on_rt_tick(self) -> None:
        """定时器回调：在没有新帧时驱动 Open3D 事件循环，保持窗口响应。"""
        if self._rt_renderer is None or not self._rt_renderer.is_open:
            if self._rt_tick_timer is not None:
                self._rt_tick_timer.stop()
            return
        still_open = self._rt_renderer.tick()
        if not still_open:
            self._log_page.append("[实时] Open3D 窗口已被关闭（tick 检测）")
            self._controller.stop_realtime_analysis()
            self._close_realtime_renderer()

    def _close_realtime_renderer(self) -> None:
        """关闭实时渲染窗口并停止定时器（幂等）。"""
        if self._rt_tick_timer is not None:
            self._rt_tick_timer.stop()
        if self._rt_renderer is not None:
            self._rt_renderer.close()
            self._rt_renderer = None
