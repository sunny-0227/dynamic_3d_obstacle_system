"""
毕业设计答辩演示版主窗口（里程碑 6）

目标：
  - 控制面板：文件/nuScenes 选择、场景/帧选择、常用按钮（加载/检测/分割/融合/一键/清空）
  - 日志面板：运行日志输出
  - 状态栏：提示当前状态
  - 控制器：后台线程执行耗时任务，UI 线程负责交互与 Open3D 显示

说明：
  - 为保证 Windows/Open3D 稳定性，Open3D 窗口仍在主线程打开（会阻塞，属正常行为）。
  - 加载/检测/分割/一键运行的计算部分放到后台线程，尽量避免 Qt 主线程卡死。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QVBoxLayout, QWidget

from app.core.fusion.result_fusion import ResultFusion
from app.core.pipeline.segment_pipeline import SegmentPipelineOutput
from app.core.segmentor.base_segmentor import SegmentationResult
from app.ui.controller import AppController, AppState
from app.ui.widgets.control_panel import ControlPanel
from app.ui.widgets.log_panel import LogPanel
from app.ui.widgets.status_bar import StatusBar
from app.utils.logger import get_logger
from app.visualization.open3d_viewer import show_pointcloud
from app.visualization.scene_renderer import RenderOptions, SceneRenderer

logger = get_logger("ui.main_window_v6")


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

        self._controller.sig_log.emit("主窗口初始化完成（里程碑6答辩演示版）")

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self) -> None:
        app_name = self._config.get("app", {}).get("name", "动态3D障碍物感知系统")
        self.setWindowTitle(app_name)
        self.setMinimumSize(980, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 8)
        layout.setSpacing(12)

        self._panel = ControlPanel()
        self._log = LogPanel(max_lines=500)
        self._status = StatusBar()
        self.setStatusBar(self._status)

        layout.addWidget(self._panel)
        layout.addWidget(self._log, stretch=1)

    def _apply_style(self) -> None:
        """风格简洁、专业，适合答辩展示。"""
        self.setStyleSheet(
            """
            QMainWindow { background-color: #1e1e2e; }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                color: #a6e3a1;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 5px;
                padding: 6px 16px;
            }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background-color: #585b70; }
            QPushButton:disabled { color: #585b70; border-color: #313244; }
            QPlainTextEdit {
                background-color: #11111b;
                border: 1px solid #313244;
                border-radius: 4px;
                color: #a6e3a1;
            }
            QComboBox, QSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 4px;
            }
            QStatusBar { background-color: #181825; color: #6c7086; }
        """
        )

    # -------------------------
    # Wiring
    # -------------------------
    def _wire(self) -> None:
        # 控制面板 -> 控制器
        self._panel.sig_select_file.connect(self._on_pick_file)
        self._panel.sig_load_pointcloud.connect(self._controller.load_current_file_pointcloud)

        self._panel.sig_select_nusc_root.connect(self._on_pick_nusc_root)
        self._panel.sig_connect_nusc.connect(self._controller.connect_nusc)
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

        # 控制器 -> UI
        self._controller.sig_log.connect(self._log.append)
        self._controller.sig_status.connect(self._status.set_status)
        self._controller.sig_error.connect(lambda t, m: QMessageBox.critical(self, t, m))
        self._controller.sig_state.connect(self._on_state)
        self._controller.sig_busy.connect(self._panel.set_busy)
        self._controller.sig_request_render.connect(self._on_render_request)

    # -------------------------
    # UI slots
    # -------------------------
    def _on_pick_file(self) -> None:
        default_dir = str(
            self._project_root
            / self._config.get("pointcloud", {}).get("default_data_dir", "data")
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择点云文件",
            default_dir,
            "点云文件 (*.bin *.pcd);;所有文件 (*.*)",
        )
        if not file_path:
            self._log.append("已取消选择文件")
            return
        p = Path(file_path)
        self._panel.set_selected_file(p)
        self._controller.set_current_file(p)
        self._status.set_status("已选择点云文件")

    def _on_pick_nusc_root(self) -> None:
        default_dir = str(self._project_root)
        picked = QFileDialog.getExistingDirectory(
            self, "选择 nuScenes mini 数据集根目录", default_dir
        )
        if not picked:
            self._log.append("已取消选择 nuScenes 根目录")
            return
        root = Path(picked)
        self._panel.set_nusc_root(root)
        self._controller.set_nusc_root(root)
        self._status.set_status("已选择 nuScenes 根目录")

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
        """
        融合显示：
          - 优先显示 last_scene（一键运行产生）
          - 否则把已有 det/seg 组装成同窗显示
          - 再不行兜底显示分割点云或原始点云
        """
        st: AppState = self._controller.state
        if st.loaded_pcd is None:
            QMessageBox.warning(self, "提示", "请先加载点云")
            return

        if st.last_scene is not None:
            self._log.append("融合显示：使用一键运行生成的融合场景（scene）")
            self._status.set_status("融合显示：scene（检测框+分割）")
            self._render_scene(st.last_scene)
            return

        pts_xyz = np.asarray(st.loaded_pcd.points, dtype=np.float32)

        try:
            has_seg = st.last_seg is not None
            has_det = bool(st.last_det) if st.last_det is not None else False

            if has_seg:
                seg_out = st.last_seg
            else:
                labels = np.zeros((pts_xyz.shape[0],), dtype=np.int32)
                seg = SegmentationResult(
                    labels=labels,
                    id_to_name={0: "background"},
                    id_to_color={0: [0.7, 0.7, 0.7]},
                )
                seg_out = SegmentPipelineOutput(points_xyz=pts_xyz, seg=seg, colored_pcd=None)

            detections = st.last_det or []
            scene = ResultFusion().fuse(points_xyz=pts_xyz, seg_out=seg_out, detections=detections)

            # 日志摘要：来源与统计，便于答辩讲解
            src = "检测+分割拼装" if (has_det and has_seg) else "仅检测" if has_det else "仅分割" if has_seg else "空结果拼装"
            self._log.append(f"融合显示：{src}（点数={pts_xyz.shape[0]:,} | 检测框={len(detections)}）")
            self._status.set_status(f"融合显示：{src}")

            if has_seg:
                labels = seg_out.seg.labels
                uniq, cnt = np.unique(labels, return_counts=True) if labels.size > 0 else ([], [])
                # 仅输出前 6 类，避免刷屏
                self._log.append("分割统计（前6类）：")
                for lid, c in list(zip(list(uniq), list(cnt)))[:6]:
                    name = seg_out.seg.id_to_name.get(int(lid), f"class_{int(lid)}")
                    self._log.append(f"  - id={int(lid):2d} | {name:12s} | 点数={int(c):,}")

            self._render_scene(scene)
            return
        except Exception as e:
            logger.warning("融合显示组装失败，将回退为简单显示：%s", e)

        if st.last_seg is not None and getattr(st.last_seg, "colored_pcd", None) is not None:
            self._log.append("融合显示：回退为仅分割点云显示（未能组装融合场景）")
            self._status.set_status("融合显示：仅分割点云（回退）")
            show_pointcloud(st.last_seg.colored_pcd, window_title="融合显示（仅分割点云）")
        else:
            self._log.append("融合显示：回退为仅原始点云显示（未能组装融合场景）")
            self._status.set_status("融合显示：仅原始点云（回退）")
            show_pointcloud(st.loaded_pcd, window_title="融合显示（仅原始点云）")

    def _on_state(self, state: AppState) -> None:
        # nuScenes 控件更新
        loader = state.nusc_loader
        if loader is not None and loader.is_connected:
            scenes = loader.get_scene_summaries()
            items = [(s["name"], s["token"]) for s in scenes]
            self._panel.set_scene_list(items)
            self._panel.set_nusc_nav_enabled(True, frame_count=loader.frame_count)
            self._update_nusc_meta()
        else:
            self._panel.set_nusc_nav_enabled(False, frame_count=0)
            self._panel.set_nusc_meta_text("未连接数据集")

        # 按钮状态：有点云才能检测/分割；有结果才能融合显示
        has_pcd = state.loaded_pcd is not None
        has_results = (
            (state.last_det is not None)
            or (state.last_seg is not None)
            or (state.last_scene is not None)
        )
        self._panel.set_pointcloud_actions_enabled(has_pcd=has_pcd, has_results=has_results)

    def _update_nusc_meta(self) -> None:
        loader = self._controller.state.nusc_loader
        if loader is None or not loader.is_connected:
            return
        n = loader.frame_count
        if n <= 0:
            self._panel.set_nusc_meta_text("当前导航下无帧")
            return
        idx = self._panel.frame_index()
        try:
            rec = loader.get_frame_record(idx)
            self._panel.set_nusc_meta_text(
                f"帧 {idx+1}/{n} | 场景: {rec.scene_name} | sample: {rec.sample_token[:8]}... | "
                f"LiDAR: {rec.lidar_path.name} | 模式: {loader.mode}"
            )
        except Exception:
            self._panel.set_nusc_meta_text(f"帧 {idx+1}/{n}（元数据解析失败）")

    def _on_render_request(self, mode: str) -> None:
        st: AppState = self._controller.state
        if mode == "seg":
            if st.last_seg is not None and getattr(st.last_seg, "colored_pcd", None) is not None:
                self._log.append("显示：语义分割结果（彩色点云）")
                self._status.set_status("显示：分割结果")
                show_pointcloud(st.last_seg.colored_pcd, window_title="语义分割结果（彩色）")
            return
        if mode == "raw":
            if st.loaded_pcd is not None:
                self._log.append("显示：原始点云预览")
                self._status.set_status("显示：点云预览")
                show_pointcloud(st.loaded_pcd, window_title="点云预览")
            return
        if mode == "fusion":
            if st.last_scene is not None:
                self._log.append("显示：融合场景（来自一键运行）")
                self._status.set_status("显示：融合场景")
                self._render_scene(st.last_scene)
            else:
                self._on_show_fusion()

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

        # Open3D 会阻塞；阻塞期间禁用面板，关闭后恢复
        self._panel.set_busy(True)
        try:
            self._renderer.render(scene)
        finally:
            self._panel.set_busy(False)

