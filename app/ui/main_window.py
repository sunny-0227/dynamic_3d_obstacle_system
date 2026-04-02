"""
PyQt5 主窗口模块
提供简洁的 GUI 界面，包含：
  - 选择点云文件 / 加载 / 运行演示（里程碑1）
  - nuScenes mini 根目录、导航与当前帧加载（里程碑2）
  - 执行检测（里程碑3：OpenPCDet 接口封装/占位实现）
  - 执行分割（里程碑4：MMDet3D 接口封装/占位实现）
  - 状态栏和日志信息区域
Open3D 采用独立弹出窗口方式，不嵌入 Qt。
"""

from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import numpy as np
import open3d as o3d

from app.datasets.nuscenes_loader import NuScenesMiniLoader
from app.datasets.nuscenes_parser import NuScenesFrameRecord
from app.core.detector.openpcdet_detector import OpenPCDetDetector
from app.core.segmentor.mmdet3d_segmentor import MMDet3DSegmentor, MMDet3DSegmentorConfig
from app.core.fusion import FusionResult, run_full_pipeline
from app.core.pipeline.detect_pipeline import DetectPipeline
from app.core.pipeline.segment_pipeline import SegmentPipeline, SegmentPipelineOutput
from app.io.pointcloud_loader import load_pointcloud
from app.utils.logger import get_logger
from app.visualization.open3d_viewer import show_fusion_result, show_pointcloud

logger = get_logger("ui.main_window")


class MainWindow(QMainWindow):
    """应用主窗口"""

    def __init__(self, config: dict = None):
        super().__init__()
        self._config = config or {}
        self._project_root = Path(__file__).resolve().parent.parent.parent

        # 当前加载的点云（来自单文件或 nuScenes）
        self._current_file: Optional[Path] = None
        self._loaded_pcd: Optional[o3d.geometry.PointCloud] = None
        self._fusion_result: Optional[FusionResult] = None
        self._last_nusc_record: Optional[NuScenesFrameRecord] = None

        # nuScenes mini
        self._nusc_dataroot: Optional[Path] = None
        self._nusc_loader: Optional[NuScenesMiniLoader] = None

        # 检测 pipeline（懒加载）
        self._detector_pipeline: Optional[DetectPipeline] = None
        self._detector: Optional[OpenPCDetDetector] = None
        self._last_detections = None  # 里程碑3：保存最近一次 DetectionBox 列表

        # 分割 pipeline（懒加载）
        self._segment_pipeline: Optional[SegmentPipeline] = None
        self._segmentor: Optional[MMDet3DSegmentor] = None
        self._last_seg_output: Optional[SegmentPipelineOutput] = None

        self._build_ui()
        self._apply_style()
        self._apply_nuscenes_config_at_startup()
        logger.info("主窗口初始化完成")

    # --------------------------------------------------
    # UI 构建
    # --------------------------------------------------
    def _build_ui(self) -> None:
        """构建窗口布局和所有控件。"""
        app_name = self._config.get("app", {}).get("name", "动态3D障碍物感知系统")
        self.setWindowTitle(app_name)
        self.setMinimumSize(820, 640)

        # 中央容器
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 8)
        root_layout.setSpacing(12)

        # ---- 标题 ----
        title_label = QLabel(app_name)
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        root_layout.addWidget(title_label)

        # ---- nuScenes mini ----
        nusc_group = QGroupBox("nuScenes mini")
        nusc_layout = QVBoxLayout(nusc_group)
        nusc_layout.setSpacing(8)

        row_root = QHBoxLayout()
        self._btn_nusc_root = QPushButton("选择数据集根目录")
        self._btn_nusc_root.setMinimumHeight(36)
        self._label_nusc_root = QLabel("未选择")
        self._label_nusc_root.setWordWrap(True)
        row_root.addWidget(self._btn_nusc_root)
        row_root.addWidget(self._label_nusc_root, stretch=1)
        nusc_layout.addLayout(row_root)

        row_conn = QHBoxLayout()
        self._btn_nusc_connect = QPushButton("加载数据集")
        self._btn_nusc_connect.setMinimumHeight(36)
        self._btn_nusc_connect.setEnabled(False)
        row_conn.addWidget(self._btn_nusc_connect)
        row_conn.addStretch()
        nusc_layout.addLayout(row_conn)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("导航方式:"))
        self._combo_nusc_mode = QComboBox()
        self._combo_nusc_mode.addItem("全数据集（sample 表顺序，非时间序）", "global")
        self._combo_nusc_mode.addItem("按场景顺序（时序链）", "scene")
        self._combo_nusc_mode.setToolTip(
            "全局：与官方 sample.json 中条目顺序一致，不等于全数据集采集时间线。\n"
            "场景：沿 first_sample → next 关键帧链，适合单段连续浏览。"
        )
        row_mode.addWidget(self._combo_nusc_mode, stretch=1)
        nusc_layout.addLayout(row_mode)

        row_scene = QHBoxLayout()
        row_scene.addWidget(QLabel("场景:"))
        self._combo_nusc_scene = QComboBox()
        self._combo_nusc_scene.setEnabled(False)
        row_scene.addWidget(self._combo_nusc_scene, stretch=1)
        nusc_layout.addLayout(row_scene)

        row_frame = QHBoxLayout()
        row_frame.addWidget(QLabel("帧索引:"))
        self._spin_nusc_frame = QSpinBox()
        self._spin_nusc_frame.setMinimum(0)
        self._spin_nusc_frame.setMaximum(0)
        self._spin_nusc_frame.setEnabled(False)
        row_frame.addWidget(self._spin_nusc_frame)
        self._btn_nusc_prev = QPushButton("上一帧")
        self._btn_nusc_next = QPushButton("下一帧")
        self._btn_nusc_load_frame = QPushButton("加载当前帧点云")
        for b in (self._btn_nusc_prev, self._btn_nusc_next, self._btn_nusc_load_frame):
            b.setMinimumHeight(36)
            b.setEnabled(False)
        row_frame.addWidget(self._btn_nusc_prev)
        row_frame.addWidget(self._btn_nusc_next)
        row_frame.addWidget(self._btn_nusc_load_frame)
        nusc_layout.addLayout(row_frame)

        self._label_nusc_meta = QLabel("请先选择并加载 nuScenes mini 根目录")
        self._label_nusc_meta.setWordWrap(True)
        nusc_layout.addWidget(self._label_nusc_meta)

        root_layout.addWidget(nusc_group)

        # ---- 文件信息区 ----
        file_group = QGroupBox("点云文件（单文件）")
        file_layout = QHBoxLayout(file_group)
        self._file_label = QLabel("未选择文件")
        self._file_label.setWordWrap(True)
        file_layout.addWidget(self._file_label, stretch=1)
        root_layout.addWidget(file_group)

        # ---- 操作按钮区 ----
        btn_group = QGroupBox("操作")
        btn_layout = QHBoxLayout(btn_group)
        btn_layout.setSpacing(10)

        self._btn_select = QPushButton("选择点云文件")
        self._btn_load = QPushButton("加载点云")
        self._btn_demo = QPushButton("运行演示")
        self._btn_detect = QPushButton("执行检测")
        self._btn_segment = QPushButton("执行分割")

        # 初始状态：加载、演示、检测按钮禁用
        self._btn_load.setEnabled(False)
        self._btn_demo.setEnabled(False)
        self._btn_detect.setEnabled(False)
        self._btn_segment.setEnabled(False)

        # 设置按钮最小高度，提升可点击性
        for btn in (self._btn_select, self._btn_load, self._btn_demo, self._btn_detect, self._btn_segment):
            btn.setMinimumHeight(40)
            btn_layout.addWidget(btn)

        root_layout.addWidget(btn_group)

        # ---- 信息展示区 ----
        info_group = QGroupBox("运行日志")
        info_layout = QVBoxLayout(info_group)
        self._log_area = QPlainTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMaximumBlockCount(200)   # 最多保留 200 行
        self._log_area.setMinimumHeight(160)
        mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        if not mono_font or mono_font.family() == "":
            mono_font = QFont("Consolas", 9)
        else:
            mono_font.setPointSize(9)
        self._log_area.setFont(mono_font)
        info_layout.addWidget(self._log_area)
        root_layout.addWidget(info_group, stretch=1)

        # ---- 状态栏 ----
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪")

        # ---- 信号连接 ----
        self._btn_select.clicked.connect(self._on_select_file)
        self._btn_load.clicked.connect(self._on_load_pointcloud)
        self._btn_demo.clicked.connect(self._on_run_demo)
        self._btn_detect.clicked.connect(self._on_execute_detection)
        self._btn_segment.clicked.connect(self._on_execute_segmentation)

        self._btn_nusc_root.clicked.connect(self._on_nusc_select_root)
        self._btn_nusc_connect.clicked.connect(self._on_nusc_connect)
        self._combo_nusc_mode.currentIndexChanged.connect(self._on_nusc_navigation_changed)
        self._combo_nusc_scene.currentIndexChanged.connect(self._on_nusc_scene_changed)
        self._btn_nusc_prev.clicked.connect(self._on_nusc_prev_frame)
        self._btn_nusc_next.clicked.connect(self._on_nusc_next_frame)
        self._btn_nusc_load_frame.clicked.connect(self._on_nusc_load_current_frame)
        self._spin_nusc_frame.valueChanged.connect(lambda _v: self._update_nusc_meta_label())

    def _apply_style(self) -> None:
        """应用简洁的深色风格样式表。"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
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
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 5px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QPushButton:pressed {
                background-color: #585b70;
            }
            QPushButton:disabled {
                color: #585b70;
                border-color: #313244;
            }
            QPlainTextEdit {
                background-color: #11111b;
                border: 1px solid #313244;
                border-radius: 4px;
                color: #a6e3a1;
            }
            QStatusBar {
                background-color: #181825;
                color: #6c7086;
            }
            QLabel {
                color: #cdd6f4;
            }
            QComboBox, QSpinBox {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 4px;
                padding: 4px;
            }
        """)

    # --------------------------------------------------
    # 交互控件批量启用/禁用（Open3D 阻塞期间）
    # --------------------------------------------------
    def _all_action_widgets(self) -> List[QWidget]:
        """需要随 Open3D 窗口同步禁用的控件列表。"""
        return [
            self._btn_select,
            self._btn_load,
            self._btn_demo,
            self._btn_detect,
            self._btn_segment,
            self._btn_nusc_root,
            self._btn_nusc_connect,
            self._combo_nusc_mode,
            self._combo_nusc_scene,
            self._spin_nusc_frame,
            self._btn_nusc_prev,
            self._btn_nusc_next,
            self._btn_nusc_load_frame,
        ]

    def _set_actions_enabled(self, enabled: bool) -> None:
        for w in self._all_action_widgets():
            w.setEnabled(enabled)
        # 根据业务状态恢复部分逻辑（在 finally 中调用 _restore_action_state）
        if enabled:
            self._restore_action_state()

    def _restore_action_state(self) -> None:
        """Open3D 关闭后，按当前数据状态恢复按钮可用性。"""
        self._btn_select.setEnabled(True)

        self._btn_nusc_root.setEnabled(True)
        self._btn_nusc_connect.setEnabled(self._nusc_dataroot is not None)

        if self._nusc_loader is not None and self._nusc_loader.is_connected:
            self._combo_nusc_mode.setEnabled(True)
            mode = self._combo_nusc_mode.currentData()
            self._combo_nusc_scene.setEnabled(mode == "scene")
            n = self._nusc_loader.frame_count
            self._spin_nusc_frame.setEnabled(n > 0)
            self._btn_nusc_prev.setEnabled(n > 0)
            self._btn_nusc_next.setEnabled(n > 0)
            self._btn_nusc_load_frame.setEnabled(n > 0)
        else:
            self._combo_nusc_mode.setEnabled(False)
            self._combo_nusc_scene.setEnabled(False)
            self._spin_nusc_frame.setEnabled(False)
            self._btn_nusc_prev.setEnabled(False)
            self._btn_nusc_next.setEnabled(False)
            self._btn_nusc_load_frame.setEnabled(False)

        if self._current_file is not None:
            self._btn_load.setEnabled(True)
        if self._loaded_pcd is not None:
            self._btn_demo.setEnabled(True)
            self._btn_detect.setEnabled(True)
            self._btn_segment.setEnabled(True)

    # --------------------------------------------------
    # nuScenes：配置中的固定根目录 / 启动时自动连接
    # --------------------------------------------------
    def _resolve_path_from_config(self, raw: str) -> Optional[Path]:
        """将配置中的路径解析为绝对 Path；无效则返回 None。"""
        if not raw or not str(raw).strip():
            return None
        p = Path(str(raw).strip())
        if not p.is_absolute():
            p = self._project_root / p
        try:
            p = p.resolve()
        except OSError:
            return None
        return p if p.is_dir() else None

    def _apply_nuscenes_config_at_startup(self) -> None:
        """
        读取 nuscenes.fixed_dataroot / auto_connect：
        在存在有效根目录与元数据子目录时预填 UI，可选延迟自动 connect。
        """
        nusc_cfg = self._config.get("nuscenes", {})
        fixed = nusc_cfg.get("fixed_dataroot", "") or ""
        resolved = self._resolve_path_from_config(fixed)
        if resolved is None:
            return

        ver = nusc_cfg.get("version", "v1.0-mini")
        meta = resolved / ver
        if not meta.is_dir():
            logger.warning(
                "配置 fixed_dataroot 已设置但缺少元数据目录 %s，已跳过自动填入: %s",
                meta.name,
                resolved,
            )
            return

        self._nusc_dataroot = resolved
        self._label_nusc_root.setText(str(resolved))
        self._btn_nusc_connect.setEnabled(True)
        self._label_nusc_meta.setText(
            "已从配置读取数据集根目录，请点击「加载数据集」"
        )
        self._log(f"已从配置读取 nuScenes 根目录: {resolved}")

        if nusc_cfg.get("auto_connect", False):
            # 进入事件循环后再连接，避免阻塞窗口构造
            QTimer.singleShot(0, self._on_nusc_connect)

    def _nusc_dialog_start_dir(self) -> str:
        """选择根目录对话框的起始路径（default_dataroot，相对路径相对项目根）。"""
        nusc_cfg = self._config.get("nuscenes", {})
        hint = nusc_cfg.get("default_dataroot", "") or ""
        if not str(hint).strip():
            return str(self._project_root)
        p = self._resolve_path_from_config(hint)
        return str(p) if p is not None else str(self._project_root)

    # --------------------------------------------------
    # 日志辅助
    # --------------------------------------------------
    def _log(self, message: str) -> None:
        """在日志区追加一行消息，并同时写入 logger。"""
        self._log_area.appendPlainText(message)
        logger.info(message)

    # --------------------------------------------------
    # nuScenes 槽函数
    # --------------------------------------------------
    def _on_nusc_select_root(self) -> None:
        """选择 nuScenes 数据集根目录（包含 samples、maps、v1.0-mini 等）。"""
        default_dir = self._nusc_dialog_start_dir()

        picked = QFileDialog.getExistingDirectory(
            self,
            "选择 nuScenes mini 数据集根目录",
            default_dir,
        )
        if not picked:
            self._log("已取消选择 nuScenes 根目录")
            return

        self._nusc_dataroot = Path(picked)
        self._nusc_loader = None
        self._label_nusc_root.setText(str(self._nusc_dataroot))
        self._btn_nusc_connect.setEnabled(True)
        self._combo_nusc_scene.clear()
        self._spin_nusc_frame.setMaximum(0)
        self._label_nusc_meta.setText("已选择根目录，请点击「加载数据集」")
        self._log(f"nuScenes 根目录: {self._nusc_dataroot}")
        self._status_bar.showMessage("已选择 nuScenes 根目录")

    def _on_nusc_connect(self) -> None:
        """实例化 NuScenes 并填充场景下拉框。"""
        if self._nusc_dataroot is None:
            QMessageBox.warning(self, "提示", "请先选择数据集根目录")
            return

        ver = self._config.get("nuscenes", {}).get("version", "v1.0-mini")
        self._status_bar.showMessage("正在连接 nuScenes ...")
        self.repaint()
        QApplication.processEvents()

        try:
            loader = NuScenesMiniLoader(self._nusc_dataroot, version=ver)
            loader.connect()
            self._nusc_loader = loader
        except Exception as exc:
            logger.error("连接 nuScenes 失败: %s", exc, exc_info=True)
            QMessageBox.critical(
                self,
                "连接失败",
                "无法加载 nuScenes：\n"
                f"{exc}\n\n"
                "说明：如果元数据不完整，会自动降级到模拟模式（扫描 samples/LIDAR_TOP/*.bin）。\n"
                "若模拟也失败，请确保你的目录中至少存在 samples/LIDAR_TOP/*.bin。",
            )
            self._status_bar.showMessage("连接失败")
            return

        self._combo_nusc_scene.blockSignals(True)
        self._combo_nusc_scene.clear()
        for s in self._nusc_loader.get_scene_summaries():
            self._combo_nusc_scene.addItem(s["name"], s["token"])
        self._combo_nusc_scene.blockSignals(False)

        mode_txt = (
            "真实模式"
            if self._nusc_loader.mode == "real"
            else "模拟模式"
            if self._nusc_loader.mode == "simulated"
            else self._nusc_loader.mode
        )
        self._log(
            f"nuScenes 连接成功 | {mode_txt} | version={ver} | 场景数={self._combo_nusc_scene.count()}"
        )
        self._on_nusc_navigation_changed()
        self._status_bar.showMessage(f"nuScenes 已连接（{mode_txt}）")

    def _current_nusc_scene_token(self) -> Optional[str]:
        if self._combo_nusc_scene.count() == 0:
            return None
        return self._combo_nusc_scene.currentData()

    def _on_nusc_navigation_changed(self) -> None:
        """切换全局/场景导航并重建帧列表。"""
        if self._nusc_loader is None or not self._nusc_loader.is_connected:
            return

        mode = self._combo_nusc_mode.currentData()
        try:
            if mode == "global":
                self._combo_nusc_scene.setEnabled(False)
                self._nusc_loader.set_navigation("global")
            else:
                self._combo_nusc_scene.setEnabled(True)
                stok = self._current_nusc_scene_token()
                if stok is None:
                    self._log("无可用场景")
                    return
                self._nusc_loader.set_navigation("scene", scene_token=stok)
        except Exception as exc:
            logger.error("重建导航失败: %s", exc, exc_info=True)
            QMessageBox.critical(self, "错误", f"导航切换失败：\n{exc}")
            return

        n = self._nusc_loader.frame_count
        self._spin_nusc_frame.blockSignals(True)
        self._spin_nusc_frame.setMaximum(max(0, n - 1))
        self._spin_nusc_frame.setValue(0)
        self._spin_nusc_frame.setEnabled(n > 0)
        self._spin_nusc_frame.blockSignals(False)
        self._btn_nusc_prev.setEnabled(n > 0)
        self._btn_nusc_next.setEnabled(n > 0)
        self._btn_nusc_load_frame.setEnabled(n > 0)
        self._update_nusc_meta_label()

    def _on_nusc_scene_changed(self) -> None:
        """场景下拉变更时，仅在「按场景」模式下重建列表。"""
        if self._combo_nusc_mode.currentData() != "scene":
            return
        self._on_nusc_navigation_changed()

    def _on_nusc_prev_frame(self) -> None:
        v = self._spin_nusc_frame.value()
        if v > self._spin_nusc_frame.minimum():
            self._spin_nusc_frame.setValue(v - 1)
        self._update_nusc_meta_label()

    def _on_nusc_next_frame(self) -> None:
        v = self._spin_nusc_frame.value()
        if v < self._spin_nusc_frame.maximum():
            self._spin_nusc_frame.setValue(v + 1)
        self._update_nusc_meta_label()

    def _update_nusc_meta_label(self) -> None:
        if self._nusc_loader is None or not self._nusc_loader.is_connected:
            self._label_nusc_meta.setText("未连接数据集")
            return
        n = self._nusc_loader.frame_count
        if n <= 0:
            self._label_nusc_meta.setText("当前导航下无帧")
            return
        idx = self._spin_nusc_frame.value()
        try:
            rec = self._nusc_loader.get_frame_record(idx)
        except Exception:
            self._label_nusc_meta.setText(f"帧 {idx + 1}/{n}（元数据解析失败）")
            return
        self._label_nusc_meta.setText(
            f"帧 {idx + 1}/{n} | 场景: {rec.scene_name} | sample: {rec.sample_token[:8]}... | LiDAR: {rec.lidar_path.name}"
        )

    def _on_nusc_load_current_frame(self) -> None:
        """从数据集层取统一帧描述，经 IO 层加载点云并预览。"""
        if self._nusc_loader is None or not self._nusc_loader.is_connected:
            QMessageBox.warning(self, "提示", "请先加载数据集")
            return

        idx = self._spin_nusc_frame.value()
        try:
            record = self._nusc_loader.get_frame_record(idx)
        except Exception as exc:
            logger.error("获取帧记录失败: %s", exc, exc_info=True)
            QMessageBox.critical(self, "错误", f"无法获取当前帧：\n{exc}")
            return

        self._last_nusc_record = record
        self._fusion_result = None
        self._last_seg_output = None
        self._current_file = record.lidar_path

        # 日志与界面一致：第 i 帧 / 共 N 帧（1-based / 总数）
        cur_1based = record.frame_index + 1
        self._log(
            f"加载 nuScenes 帧 | 模式={record.navigation_mode} | "
            f"{record.scene_name} | 第 {cur_1based}/{record.frame_count} 帧"
        )
        self._log(f"LiDAR 路径: {record.lidar_path}")
        self._status_bar.showMessage("正在加载 LiDAR ...")
        self.repaint()

        if not record.lidar_path.is_file():
            QMessageBox.critical(
                self,
                "文件缺失",
                f"点云文件不存在：\n{record.lidar_path}\n请确认已下载对应样本。",
            )
            self._status_bar.showMessage("文件缺失")
            return

        try:
            self._loaded_pcd = load_pointcloud(record.lidar_path)
            n_pts = len(self._loaded_pcd.points)
            self._file_label.setText(str(record.lidar_path))
            self._log(f"加载成功！点云共 {n_pts:,} 个点")
            self._btn_demo.setEnabled(True)
            self._btn_detect.setEnabled(True)
            self._btn_segment.setEnabled(True)
            self._btn_load.setEnabled(True)
            self._status_bar.showMessage(
                f"已加载 nuScenes 帧 {idx + 1}/{record.frame_count}"
            )
            self._log("正在打开 Open3D 预览窗口（关闭窗口后可继续操作）...")
            self._launch_viewer("raw")
        except Exception as exc:
            logger.error("加载 nuScenes 点云失败: %s", exc, exc_info=True)
            QMessageBox.critical(self, "加载失败", f"加载点云时发生错误：\n{exc}")
            self._status_bar.showMessage("加载失败")

    # --------------------------------------------------
    # 槽函数：选择文件
    # --------------------------------------------------
    def _on_select_file(self) -> None:
        """弹出文件对话框，让用户选择 .bin 或 .pcd 文件。"""
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
            self._log("已取消选择文件")
            return

        self._current_file = Path(file_path)
        self._loaded_pcd = None
        self._fusion_result = None
        self._last_seg_output = None
        self._last_nusc_record = None
        self._btn_load.setEnabled(True)
        self._btn_demo.setEnabled(False)
        self._btn_detect.setEnabled(False)
        self._btn_segment.setEnabled(False)

        self._file_label.setText(str(self._current_file))
        self._log(f"已选择文件: {self._current_file}")
        self._status_bar.showMessage(f"已选择: {self._current_file.name}")

    # --------------------------------------------------
    # 槽函数：加载点云
    # --------------------------------------------------
    def _on_load_pointcloud(self) -> None:
        """加载当前选中的点云文件并显示基本信息。"""
        if self._current_file is None:
            QMessageBox.warning(self, "警告", "请先选择点云文件！")
            return

        self._log(f"正在加载: {self._current_file.name} ...")
        self._status_bar.showMessage("加载中...")
        self.repaint()  # 强制刷新界面，显示"加载中"状态

        try:
            self._loaded_pcd = load_pointcloud(self._current_file)
            n_pts = len(self._loaded_pcd.points)
            self._log(f"加载成功！点云共 {n_pts:,} 个点")
            self._status_bar.showMessage(f"已加载 {n_pts:,} 个点")
            self._btn_demo.setEnabled(True)
            self._btn_detect.setEnabled(True)
            self._btn_segment.setEnabled(True)

            # 加载后立即弹窗预览原始点云
            self._log("正在打开 Open3D 预览窗口（关闭窗口后可继续操作）...")
            self._launch_viewer("raw")

        except Exception as exc:
            logger.error("加载点云失败: %s", exc, exc_info=True)
            QMessageBox.critical(self, "加载失败", f"加载点云时发生错误：\n{exc}")
            self._status_bar.showMessage("加载失败")

    # --------------------------------------------------
    # 槽函数：运行演示
    # --------------------------------------------------
    def _on_run_demo(self) -> None:
        """执行伪推理流水线并在 Open3D 窗口显示结果。"""
        if self._loaded_pcd is None:
            QMessageBox.warning(self, "警告", "请先加载点云！")
            return

        self._log("开始运行演示（伪分割 + 伪检测）...")
        self._status_bar.showMessage("推理中...")
        self.repaint()

        try:
            det_cfg = self._config.get("fake_detector", {})
            seg_cfg = self._config.get("fake_segmentor", {})

            num_boxes = det_cfg.get("num_boxes", 3)
            score_range = tuple(det_cfg.get("score_range", [0.75, 0.99]))
            num_classes = seg_cfg.get("num_classes", 4)
            # 从配置中读取类别颜色，None 时 run_full_pipeline 内部使用默认值
            class_colors = seg_cfg.get("class_colors", None)

            self._fusion_result = run_full_pipeline(
                self._loaded_pcd,
                num_boxes=num_boxes,
                score_range=score_range,
                num_classes=num_classes,
                class_colors=class_colors,
            )

            # 打印检测摘要
            self._log(f"演示完成！生成 {len(self._fusion_result.detections)} 个检测框：")
            for i, det in enumerate(self._fusion_result.detections):
                self._log(
                    f"  #{i+1} 类别: {det.label:12s} | 置信度: {det.score:.3f}"
                )

            self._status_bar.showMessage("演示完成，Open3D 窗口已打开")
            self._log("正在打开 Open3D 融合结果窗口（关闭窗口后可继续操作）...")
            self._launch_viewer("fusion")

        except Exception as exc:
            logger.error("运行演示失败: %s", exc, exc_info=True)
            QMessageBox.critical(self, "演示失败", f"运行演示时发生错误：\n{exc}")
            self._status_bar.showMessage("演示失败")

    # --------------------------------------------------
    # 启动 Open3D 窗口（主线程直接调用）
    # --------------------------------------------------
    def _on_execute_detection(self) -> None:
        """对当前加载的点云执行真实/占位检测（里程碑 3）。"""
        if self._loaded_pcd is None:
            QMessageBox.warning(self, "警告", "请先加载点云（或加载 nuScenes 帧点云）！")
            return

        if self._detector_pipeline is None:
            # 从配置加载检测器参数（当前主要用于占位 fallback，真实接入后可在 OpenPCDetDetector 内实现）
            det_cfg = self._config.get("detector", {})
            score_threshold = det_cfg.get("score_threshold", 0.1)
            num_boxes_fake = det_cfg.get("num_boxes_fake", 3)
            class_names = det_cfg.get("class_names", ["car", "pedestrian", "cyclist"])

            openpcdet_cfg = det_cfg.get("openpcdet", {})
            model_cfg_raw = openpcdet_cfg.get("model_cfg", "") or ""
            ckpt_raw = openpcdet_cfg.get("checkpoint_path", "") or ""
            device = openpcdet_cfg.get("device", "cpu")

            def _resolve_maybe_relative(p: str) -> Optional[Path]:
                if not p or not str(p).strip():
                    return None
                pp = Path(str(p).strip())
                if not pp.is_absolute():
                    pp = self._project_root / pp
                return pp

            model_cfg_path = _resolve_maybe_relative(model_cfg_raw)
            checkpoint_path = _resolve_maybe_relative(ckpt_raw)

            self._detector = OpenPCDetDetector(
                model_cfg_path=model_cfg_path,
                checkpoint_path=checkpoint_path,
                device=device,
                score_threshold=score_threshold,
                num_boxes_fake=num_boxes_fake,
                class_names=class_names,
            )
            self._detector_pipeline = DetectPipeline(detector=self._detector)

        # 检测期间禁用按钮，避免重复点击
        self._btn_detect.setEnabled(False)
        QApplication.processEvents()

        try:
            points_xyz = np.asarray(self._loaded_pcd.points, dtype=np.float32)
            self._log("开始执行检测（里程碑 3：统一接口）...")
            self._status_bar.showMessage("检测中...")

            det_boxes = self._detector_pipeline.run(points_xyz)
            self._last_detections = det_boxes

            self._log(f"检测完成！共 {len(det_boxes)} 个检测框：")
            for i, box in enumerate(det_boxes):
                cx, cy, cz = box.center.tolist()
                l, w, h = box.size.tolist()
                self._log(
                    f"  #{i+1} 类别: {box.class_name:12s} | score={box.score:.3f} | "
                    f"center=({cx:.2f},{cy:.2f},{cz:.2f}) | size=({l:.2f},{w:.2f},{h:.2f}) | yaw={box.yaw:.3f}"
                )
            self._status_bar.showMessage("检测完成（已输出到日志区）")
        except Exception as exc:
            logger.error("执行检测失败: %s", exc, exc_info=True)
            QMessageBox.critical(self, "检测失败", f"执行检测时发生错误：\n{exc}")
            self._status_bar.showMessage("检测失败")
        finally:
            # 恢复按钮可用性（只要点云存在即可执行）
            self._btn_detect.setEnabled(self._loaded_pcd is not None)

    def _on_execute_segmentation(self) -> None:
        """对当前加载的点云执行真实/占位语义分割（里程碑 4）。"""
        if self._loaded_pcd is None:
            QMessageBox.warning(self, "警告", "请先加载点云（或加载 nuScenes 帧点云）！")
            return

        if self._segment_pipeline is None:
            seg_cfg = self._config.get("segmentor", {})
            backend = seg_cfg.get("backend", "mmdet3d")
            if backend != "mmdet3d":
                logger.warning("未知分割后端 %s，当前仅提供 mmdet3d 占位封装，已继续使用 mmdet3d。", backend)

            mmd = seg_cfg.get("mmdet3d", {})
            num_classes = int(seg_cfg.get("num_classes", 4))
            class_names = seg_cfg.get("class_names", None)
            palette = seg_cfg.get("palette", None)
            if palette is None:
                # 尝试复用里程碑1的假分割颜色配置，若存在
                palette = self._config.get("fake_segmentor", {}).get("class_colors", None)

            cfg_obj = MMDet3DSegmentorConfig(
                config_file=str(mmd.get("config_file", "") or ""),
                checkpoint_file=str(mmd.get("checkpoint_file", "") or ""),
                device=str(mmd.get("device", "cpu") or "cpu"),
                num_classes=num_classes,
                class_names=class_names,
                palette=palette,
            )
            self._segmentor = MMDet3DSegmentor(cfg_obj)
            self._segment_pipeline = SegmentPipeline(segmentor=self._segmentor)

        self._btn_segment.setEnabled(False)
        QApplication.processEvents()

        try:
            points_xyz = np.asarray(self._loaded_pcd.points, dtype=np.float32)
            self._log("开始执行分割（里程碑 4：统一接口）...")
            self._status_bar.showMessage("分割中...")

            out = self._segment_pipeline.run(points_xyz)
            self._last_seg_output = out

            # 输出简单摘要
            labels = out.seg.labels
            uniq, cnt = np.unique(labels, return_counts=True) if labels.size > 0 else ([], [])
            self._log(f"分割完成！共 {labels.shape[0]:,} 个点，类别统计：")
            for lid, c in zip(list(uniq), list(cnt)):
                name = out.seg.id_to_name.get(int(lid), f"class_{int(lid)}")
                self._log(f"  - id={int(lid):2d} | {name:12s} | 点数={int(c):,}")

            if out.colored_pcd is not None:
                self._log("正在打开 Open3D 分割彩色点云窗口（关闭窗口后可继续操作）...")
                self._status_bar.showMessage("分割完成，Open3D 窗口已打开")
                self._launch_viewer("seg")
            else:
                self._status_bar.showMessage("分割完成（未生成 Open3D 彩色点云）")
        except Exception as exc:
            logger.error("执行分割失败: %s", exc, exc_info=True)
            QMessageBox.critical(self, "分割失败", f"执行分割时发生错误：\n{exc}")
            self._status_bar.showMessage("分割失败")
        finally:
            self._btn_segment.setEnabled(self._loaded_pcd is not None)

    def _launch_viewer(self, mode: str) -> None:
        """
        在主线程直接调用 Open3D 窗口。
        Open3D 底层依赖 GLFW/OpenGL，Windows 要求 OpenGL 上下文必须在
        创建它的线程中使用（即主线程），在子线程中调用会导致崩溃或黑屏。
        Open3D vis.run() 会阻塞直到用户关闭窗口，关闭后 Qt 自动恢复响应。
        """
        vis_cfg = self._config.get("visualization", {})
        bg = vis_cfg.get("background_color", [0.05, 0.05, 0.05])
        w = vis_cfg.get("window_width", 1280)
        h = vis_cfg.get("window_height", 720)
        ps = vis_cfg.get("point_size", 2.0)
        title = vis_cfg.get("window_title", "3D点云感知结果")

        # 打开窗口前禁用所有操作控件，防止重复点击
        self._set_actions_enabled(False)
        QApplication.processEvents()

        try:
            if mode == "raw" and self._loaded_pcd is not None:
                show_pointcloud(
                    self._loaded_pcd,
                    window_title="点云预览",
                    width=w, height=h,
                    background_color=bg,
                    point_size=ps,
                )
            elif mode == "seg" and self._last_seg_output is not None and self._last_seg_output.colored_pcd is not None:
                show_pointcloud(
                    self._last_seg_output.colored_pcd,
                    window_title="语义分割结果（彩色）",
                    width=w, height=h,
                    background_color=bg,
                    point_size=ps,
                )
            elif mode == "fusion" and self._fusion_result is not None:
                show_fusion_result(
                    self._fusion_result,
                    window_title=title,
                    width=w, height=h,
                    background_color=bg,
                    point_size=ps,
                )
        except Exception as exc:
            logger.error("Open3D 渲染错误: %s", exc, exc_info=True)
            QMessageBox.critical(self, "渲染错误", f"Open3D 窗口发生错误：\n{exc}")
        finally:
            self._set_actions_enabled(True)
            self._status_bar.showMessage("就绪")
            self._log("Open3D 窗口已关闭，可继续操作")
            self._update_nusc_meta_label()
