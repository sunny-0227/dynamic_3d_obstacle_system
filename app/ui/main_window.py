"""
PyQt5 主窗口模块
提供简洁的 GUI 界面，包含：
  - 选择点云文件按钮
  - 加载点云按钮
  - 运行演示按钮（伪分割+伪检测+Open3D 弹窗）
  - 状态栏和日志信息区域
Open3D 采用独立弹出窗口方式，不嵌入 Qt。
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import open3d as o3d

from app.io.pointcloud_loader import load_pointcloud
from app.core.fusion import run_full_pipeline, FusionResult
from app.visualization.open3d_viewer import show_pointcloud, show_fusion_result
from app.utils.logger import get_logger

logger = get_logger("ui.main_window")


# -------------------------------------------------------
# 主窗口
# -------------------------------------------------------
class MainWindow(QMainWindow):
    """应用主窗口"""

    def __init__(self, config: dict = None):
        super().__init__()
        self._config = config or {}
        self._current_file: Optional[Path] = None
        self._loaded_pcd: Optional[o3d.geometry.PointCloud] = None
        self._fusion_result: Optional[FusionResult] = None

        self._build_ui()
        self._apply_style()
        logger.info("主窗口初始化完成")

    # --------------------------------------------------
    # UI 构建
    # --------------------------------------------------
    def _build_ui(self) -> None:
        """构建窗口布局和所有控件。"""
        app_name = self._config.get("app", {}).get("name", "动态3D障碍物感知系统")
        self.setWindowTitle(app_name)
        self.setMinimumSize(700, 480)

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

        # ---- 文件信息区 ----
        file_group = QGroupBox("点云文件")
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

        # 初始状态：加载和演示按钮禁用
        self._btn_load.setEnabled(False)
        self._btn_demo.setEnabled(False)

        # 设置按钮最小高度，提升可点击性
        for btn in (self._btn_select, self._btn_load, self._btn_demo):
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
        mono_font = QFont("Consolas", 9)
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
        """)

    # --------------------------------------------------
    # 日志辅助
    # --------------------------------------------------
    def _log(self, message: str) -> None:
        """在日志区追加一行消息，并同时写入 logger。"""
        self._log_area.appendPlainText(message)
        logger.info(message)

    # --------------------------------------------------
    # 槽函数：选择文件
    # --------------------------------------------------
    def _on_select_file(self) -> None:
        """弹出文件对话框，让用户选择 .bin 或 .pcd 文件。"""
        default_dir = str(
            Path(__file__).resolve().parent.parent.parent
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
        self._btn_load.setEnabled(True)
        self._btn_demo.setEnabled(False)

        display_path = self._current_file.name
        self._file_label.setText(str(self._current_file))
        self._log(f"已选择文件: {self._current_file}")
        self._status_bar.showMessage(f"已选择: {display_path}")

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

            self._fusion_result = run_full_pipeline(
                self._loaded_pcd,
                num_boxes=num_boxes,
                score_range=score_range,
                num_classes=num_classes,
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

        # 打开窗口前禁用按钮，防止用户在窗口打开期间重复点击
        self._btn_select.setEnabled(False)
        self._btn_load.setEnabled(False)
        self._btn_demo.setEnabled(False)
        # 强制刷新 Qt 界面，让按钮禁用状态立即生效
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
            # 窗口关闭后恢复按钮状态
            self._btn_select.setEnabled(True)
            if self._current_file:
                self._btn_load.setEnabled(True)
            if self._loaded_pcd is not None:
                self._btn_demo.setEnabled(True)
            self._status_bar.showMessage("就绪")
            self._log("Open3D 窗口已关闭，可继续操作")
