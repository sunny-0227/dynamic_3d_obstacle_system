from __future__ import annotations

"""
实时相机感知页面

包含：
  - 数据源选择（RealSense / Mock）
  - 实时流目录选择（Mock 模式）
  - 启动/停止实时模式
  - 开始/停止实时分析
  - 实时状态仪表盘（FPS、点数、障碍物数、分割状态、检测状态、相机连接状态）
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class RealtimePage(QWidget):
    """实时相机感知页面，所有信号在此定义，由 MainWindow 连接到 Controller。"""

    sig_camera_source_changed    = pyqtSignal(str)   # "realsense" | "mock"
    sig_select_realtime_dir      = pyqtSignal()
    sig_start_realtime           = pyqtSignal()
    sig_stop_realtime            = pyqtSignal()
    sig_start_realtime_analysis  = pyqtSignal()
    sig_stop_realtime_analysis   = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_internal()

    # ------------------------------------------------------------------
    # 构建 UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        title = QLabel("实时相机感知")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(self._make_hline())

        body = QHBoxLayout()
        body.setSpacing(16)
        body.addLayout(self._build_left_col(), stretch=1)
        body.addLayout(self._build_right_col(), stretch=1)
        root.addLayout(body)

        root.addWidget(self._make_hline())
        root.addWidget(self._build_status_panel())
        root.addStretch(1)

    def _build_left_col(self) -> QVBoxLayout:
        """左列：数据源 + 目录选择 + 启停控制。"""
        col = QVBoxLayout()
        col.setSpacing(12)

        # ── 数据源 ────────────────────────────────────────────────
        src_group = QGroupBox("数据源选择")
        sg_lay = QVBoxLayout(src_group)
        sg_lay.setSpacing(8)

        self._radio_realsense = QRadioButton("Intel RealSense 深度相机")
        self._radio_mock      = QRadioButton("Mock 点云流（本地文件循环）")
        self._btn_grp_cam     = QButtonGroup(self)
        self._btn_grp_cam.addButton(self._radio_realsense, 0)
        self._btn_grp_cam.addButton(self._radio_mock,      1)
        self._radio_realsense.setChecked(True)

        sg_lay.addWidget(self._radio_realsense)
        sg_lay.addWidget(self._radio_mock)

        hint_src = QLabel("RealSense 模式直接连接硬件；Mock 模式从本地目录循环读取点云文件。")
        hint_src.setObjectName("hintLabel")
        hint_src.setWordWrap(True)
        sg_lay.addWidget(hint_src)
        col.addWidget(src_group)

        # ── Mock 目录 ─────────────────────────────────────────────
        self._mock_group = QGroupBox("Mock 点云流目录")
        mg_lay = QHBoxLayout(self._mock_group)
        self._btn_pick_dir = QPushButton("选择目录")
        self._lbl_rt_dir   = QLineEdit()
        self._lbl_rt_dir.setReadOnly(True)
        self._lbl_rt_dir.setPlaceholderText("未选择目录")
        mg_lay.addWidget(self._btn_pick_dir)
        mg_lay.addWidget(self._lbl_rt_dir, stretch=1)
        self._mock_group.setEnabled(False)   # 默认 RealSense 模式，Mock 目录隐藏
        col.addWidget(self._mock_group)

        # ── 模式控制 ──────────────────────────────────────────────
        ctrl_group = QGroupBox("实时模式控制")
        cl = QVBoxLayout(ctrl_group)
        cl.setSpacing(10)

        self._btn_start_rt  = QPushButton("▶  启动实时模式")
        self._btn_stop_rt   = QPushButton("■  停止实时模式")
        self._btn_start_rt.setMinimumHeight(40)
        self._btn_stop_rt.setMinimumHeight(40)
        self._btn_stop_rt.setEnabled(False)
        cl.addWidget(self._btn_start_rt)
        cl.addWidget(self._btn_stop_rt)

        cl.addWidget(self._make_hline())

        self._btn_start_ana = QPushButton("▶▶  开始实时分析")
        self._btn_stop_ana  = QPushButton("■■  停止实时分析")
        self._btn_start_ana.setMinimumHeight(40)
        self._btn_stop_ana.setMinimumHeight(40)
        self._btn_start_ana.setEnabled(False)
        self._btn_stop_ana.setEnabled(False)
        cl.addWidget(self._btn_start_ana)
        cl.addWidget(self._btn_stop_ana)

        col.addWidget(ctrl_group)
        col.addStretch(1)
        return col

    def _build_right_col(self) -> QVBoxLayout:
        """右列：实时仪表盘。"""
        col = QVBoxLayout()
        col.setSpacing(12)

        dash_group = QGroupBox("实时状态仪表盘")
        dl = QVBoxLayout(dash_group)
        dl.setSpacing(10)

        self._lbl_cam_status = self._make_status_row(dl, "相机连接状态", "未连接")
        self._lbl_fps        = self._make_status_row(dl, "当前 FPS",    "—")
        self._lbl_points     = self._make_status_row(dl, "当前点数",    "—")
        self._lbl_obstacles  = self._make_status_row(dl, "当前障碍物数", "—")
        self._lbl_seg_status = self._make_status_row(dl, "实时分割状态", "未运行")
        self._lbl_det_status = self._make_status_row(dl, "实时检测框状态", "未运行")
        self._lbl_source     = self._make_status_row(dl, "数据来源",    "—")

        col.addWidget(dash_group)

        # 提示区
        tip_group = QGroupBox("操作说明")
        tl = QVBoxLayout(tip_group)
        tips = [
            "1. 选择数据源（RealSense / Mock）",
            "2. 点击「启动实时模式」初始化相机",
            "3. 相机就绪后，点击「开始实时分析」",
            "4. Open3D 窗口将自动弹出显示三维结果",
            "5. 点击「停止实时分析」或关闭 Open3D 窗口结束",
        ]
        for t in tips:
            lbl = QLabel(t)
            lbl.setObjectName("hintLabel")
            tl.addWidget(lbl)

        col.addWidget(tip_group)
        col.addStretch(1)
        return col

    def _build_status_panel(self) -> QWidget:
        panel = QGroupBox("实时会话信息")
        lay = QHBoxLayout(panel)

        self._lbl_session_fps    = QLabel("FPS: —")
        self._lbl_session_pts    = QLabel("点数: —")
        self._lbl_session_obs    = QLabel("障碍物: —")
        self._lbl_session_mode   = QLabel("数据源: —")

        for lbl in (self._lbl_session_fps, self._lbl_session_pts,
                    self._lbl_session_obs, self._lbl_session_mode):
            lbl.setObjectName("statusLabel")
            lay.addWidget(lbl, stretch=1)

        return panel

    @staticmethod
    def _make_status_row(parent_lay: QVBoxLayout, label: str, default: str) -> QLabel:
        row = QHBoxLayout()
        key = QLabel(f"{label}：")
        key.setFixedWidth(130)
        key.setObjectName("statusKey")
        val = QLabel(default)
        val.setObjectName("statusValue")
        row.addWidget(key)
        row.addWidget(val, stretch=1)
        parent_lay.addLayout(row)
        return val

    @staticmethod
    def _make_hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setObjectName("hline")
        return line

    # ------------------------------------------------------------------
    # 内部信号连接
    # ------------------------------------------------------------------

    def _connect_internal(self) -> None:
        self._btn_grp_cam.buttonClicked.connect(self._on_cam_changed)
        self._btn_pick_dir.clicked.connect(self.sig_select_realtime_dir.emit)
        self._btn_start_rt.clicked.connect(self.sig_start_realtime.emit)
        self._btn_stop_rt.clicked.connect(self.sig_stop_realtime.emit)
        self._btn_start_ana.clicked.connect(self.sig_start_realtime_analysis.emit)
        self._btn_stop_ana.clicked.connect(self.sig_stop_realtime_analysis.emit)

    def _on_cam_changed(self) -> None:
        src = "mock" if self._radio_mock.isChecked() else "realsense"
        self._mock_group.setEnabled(src == "mock")
        self.sig_camera_source_changed.emit(src)

    # ------------------------------------------------------------------
    # 公开接口（供 MainWindow 调用）
    # ------------------------------------------------------------------

    def camera_source(self) -> str:
        return "mock" if self._radio_mock.isChecked() else "realsense"

    def set_realtime_stream_dir(self, p: Optional[Path]) -> None:
        self._lbl_rt_dir.setText(str(p) if p else "")

    def set_controls(self, running: bool, analyzing: bool) -> None:
        """根据运行/分析状态切换按钮可用性。"""
        self._btn_start_rt.setEnabled(not running)
        self._btn_stop_rt.setEnabled(running)
        self._btn_start_ana.setEnabled(running and not analyzing)
        self._btn_stop_ana.setEnabled(analyzing)

    def set_stats(
        self,
        fps: float,
        points: int,
        obstacles: int,
        source: str,
        running: bool,
        analyzing: bool,
    ) -> None:
        """刷新仪表盘所有数值。"""
        cam_txt = "已连接 / 运行中" if running else "未连接"
        self._lbl_cam_status.setText(cam_txt)
        self._lbl_fps.setText(f"{fps:.1f}" if running else "—")
        self._lbl_points.setText(f"{points:,}" if running else "—")
        self._lbl_obstacles.setText(f"{obstacles}" if analyzing else "—")
        self._lbl_seg_status.setText("分析中" if analyzing else ("就绪" if running else "未运行"))
        self._lbl_det_status.setText("检测中" if analyzing else ("就绪" if running else "未运行"))
        self._lbl_source.setText(source if source else "—")

        # 底部摘要行
        self._lbl_session_fps.setText(f"FPS: {fps:.1f}" if running else "FPS: —")
        self._lbl_session_pts.setText(f"点数: {points:,}" if running else "点数: —")
        self._lbl_session_obs.setText(f"障碍物: {obstacles}" if analyzing else "障碍物: —")
        self._lbl_session_mode.setText(f"数据源: {source}" if source else "数据源: —")

    def set_busy(self, busy: bool) -> None:
        # busy 期间只禁用"启动"类按钮；"停止"类按钮保持可用，确保用户随时可以中断
        self._btn_start_rt.setEnabled(not busy)
        self._btn_start_ana.setEnabled(not busy)
