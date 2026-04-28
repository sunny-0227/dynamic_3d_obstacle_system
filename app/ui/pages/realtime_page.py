from __future__ import annotations

"""
实时相机感知页面

包含：
  - 数据源选择（RealSense / Mock）
  - 实时流目录选择（Mock 模式）
  - 启动/停止实时模式
  - 开始/停止实时分析
  - 实时状态仪表盘（FPS、点数、障碍物数、分割状态、检测状态、相机连接状态）

布局优化记录：
  - 外边距 28 24（左右/上下）
  - GroupBox 间距 18px，内部 spacing=10，内边距 14px
  - 控制按钮高度：40→46px
  - 仪表盘 key 标签宽度 140px，对齐整洁
  - 左右列 stretch=1:1
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
        # 【间距优化】外边距 28 左右、24 上下
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)              # 【间距优化】各行间距 18px

        title = QLabel("实时相机感知")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(self._make_hline())

        body = QHBoxLayout()
        body.setSpacing(24)              # 【间距优化】左右列间距 24px
        body.addLayout(self._build_left_col(), stretch=1)    # 【拉伸】左列 1
        body.addLayout(self._build_right_col(), stretch=1)   # 【拉伸】右列 1
        root.addLayout(body, stretch=1)  # 【拉伸】body 占满剩余高度

        root.addWidget(self._make_hline())
        root.addWidget(self._build_status_panel())

    def _build_left_col(self) -> QVBoxLayout:
        """左列：数据源 + 目录选择 + 启停控制。"""
        col = QVBoxLayout()
        col.setSpacing(18)               # 【间距优化】GroupBox 间距 18px

        # ── 数据源 ────────────────────────────────────────────────
        src_group = QGroupBox("数据源选择")
        sg_lay = QVBoxLayout(src_group)
        sg_lay.setSpacing(10)
        sg_lay.setContentsMargins(14, 18, 14, 14)

        self._radio_realsense = QRadioButton("Intel RealSense 深度相机")
        self._radio_mock      = QRadioButton("Mock 点云流（本地文件循环）")
        self._btn_grp_cam     = QButtonGroup(self)
        self._btn_grp_cam.addButton(self._radio_realsense, 0)
        self._btn_grp_cam.addButton(self._radio_mock,      1)
        self._radio_realsense.setChecked(True)

        sg_lay.addWidget(self._radio_realsense)
        sg_lay.addWidget(self._radio_mock)

        hint_src = QLabel("RealSense 直接连接硬件；Mock 从本地目录循环读取点云文件。")
        hint_src.setObjectName("hintLabel")
        hint_src.setWordWrap(True)
        sg_lay.addWidget(hint_src)
        col.addWidget(src_group)

        # ── Mock 目录 ─────────────────────────────────────────────
        self._mock_group = QGroupBox("Mock 点云流目录")
        mg_lay = QHBoxLayout(self._mock_group)
        mg_lay.setSpacing(10)
        mg_lay.setContentsMargins(14, 18, 14, 14)

        self._btn_pick_dir = QPushButton("选择目录")
        self._btn_pick_dir.setMinimumHeight(34)
        self._btn_pick_dir.setFixedWidth(90)
        self._lbl_rt_dir   = QLineEdit()
        self._lbl_rt_dir.setReadOnly(True)
        self._lbl_rt_dir.setPlaceholderText("未选择目录")
        self._lbl_rt_dir.setMinimumWidth(200)    # 【最小宽度】防止被压缩
        self._lbl_rt_dir.setMinimumHeight(34)
        mg_lay.addWidget(self._btn_pick_dir)
        mg_lay.addWidget(self._lbl_rt_dir, stretch=1)
        self._mock_group.setEnabled(False)
        col.addWidget(self._mock_group)

        # ── 模式控制 ──────────────────────────────────────────────
        ctrl_group = QGroupBox("实时模式控制")
        cl = QVBoxLayout(ctrl_group)
        cl.setSpacing(12)                # 【间距优化】按钮行间距 12px
        cl.setContentsMargins(14, 18, 14, 16)

        self._btn_start_rt  = QPushButton("▶  启动实时模式")
        self._btn_stop_rt   = QPushButton("■  停止实时模式")
        self._btn_start_rt.setMinimumHeight(46)   # 【间距优化】控制按钮高 46px
        self._btn_stop_rt.setMinimumHeight(46)
        self._btn_stop_rt.setEnabled(False)
        cl.addWidget(self._btn_start_rt)
        cl.addWidget(self._btn_stop_rt)

        cl.addSpacing(4)
        cl.addWidget(self._make_hline())
        cl.addSpacing(4)

        self._btn_start_ana = QPushButton("▶▶  开始实时分析")
        self._btn_stop_ana  = QPushButton("■■  停止实时分析")
        self._btn_start_ana.setMinimumHeight(46)
        self._btn_stop_ana.setMinimumHeight(46)
        self._btn_start_ana.setEnabled(False)
        self._btn_stop_ana.setEnabled(False)
        cl.addWidget(self._btn_start_ana)
        cl.addWidget(self._btn_stop_ana)

        col.addWidget(ctrl_group)
        col.addStretch(1)                # 【拉伸】左列底部留白
        return col

    def _build_right_col(self) -> QVBoxLayout:
        """右列：实时仪表盘 + 操作说明。"""
        col = QVBoxLayout()
        col.setSpacing(18)

        dash_group = QGroupBox("实时状态仪表盘")
        dl = QVBoxLayout(dash_group)
        dl.setSpacing(12)                # 【间距优化】仪表盘行间距 12px
        dl.setContentsMargins(14, 20, 14, 16)

        self._lbl_cam_status = self._make_status_row(dl, "相机连接状态", "未连接")
        self._lbl_fps        = self._make_status_row(dl, "当前 FPS",     "—")
        self._lbl_points     = self._make_status_row(dl, "当前点数",     "—")
        self._lbl_obstacles  = self._make_status_row(dl, "当前障碍物数", "—")
        self._lbl_seg_status = self._make_status_row(dl, "实时分割状态", "未运行")
        self._lbl_det_status = self._make_status_row(dl, "实时检测框状态", "未运行")
        self._lbl_source     = self._make_status_row(dl, "数据来源",     "—")

        col.addWidget(dash_group)

        # 提示区
        tip_group = QGroupBox("操作说明")
        tl = QVBoxLayout(tip_group)
        tl.setSpacing(8)
        tl.setContentsMargins(14, 18, 14, 14)
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
            lbl.setMinimumHeight(22)      # 【间距优化】每条说明有最小高度
            tl.addWidget(lbl)

        col.addWidget(tip_group)
        col.addStretch(1)
        return col

    def _build_status_panel(self) -> QWidget:
        """底部实时会话摘要行。"""
        panel = QGroupBox("实时会话信息")
        lay = QHBoxLayout(panel)
        lay.setSpacing(24)               # 【间距优化】各栏间距 24px
        lay.setContentsMargins(16, 14, 16, 14)

        self._lbl_session_fps    = QLabel("FPS: —")
        self._lbl_session_pts    = QLabel("点数: —")
        self._lbl_session_obs    = QLabel("障碍物: —")
        self._lbl_session_mode   = QLabel("数据源: —")

        for lbl in (self._lbl_session_fps, self._lbl_session_pts,
                    self._lbl_session_obs, self._lbl_session_mode):
            lbl.setObjectName("statusLabel")
            lbl.setMinimumHeight(22)
            lay.addWidget(lbl, stretch=1)  # 【拉伸】四列均等

        return panel

    @staticmethod
    def _make_status_row(parent_lay: QVBoxLayout, label: str, default: str) -> QLabel:
        """创建一行 key: value 状态显示，key 标签宽度固定确保对齐。"""
        row = QHBoxLayout()
        row.setSpacing(12)               # 【间距优化】key 与 value 间距 12px
        key = QLabel(f"{label}：")
        key.setFixedWidth(140)           # 【间距优化】key 宽度统一 140px（从 130 放宽）
        key.setObjectName("statusKey")
        val = QLabel(default)
        val.setObjectName("statusValue")
        val.setMinimumHeight(24)
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
    # 内部信号连接（不改变）
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
    # 公开接口（不改变）
    # ------------------------------------------------------------------

    def camera_source(self) -> str:
        return "mock" if self._radio_mock.isChecked() else "realsense"

    def set_realtime_stream_dir(self, p: Optional[Path]) -> None:
        self._lbl_rt_dir.setText(str(p) if p else "")

    def set_controls(self, running: bool, analyzing: bool) -> None:
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
        cam_txt = "已连接 / 运行中" if running else "未连接"
        self._lbl_cam_status.setText(cam_txt)
        self._lbl_fps.setText(f"{fps:.1f}" if running else "—")
        self._lbl_points.setText(f"{points:,}" if running else "—")
        self._lbl_obstacles.setText(f"{obstacles}" if analyzing else "—")
        self._lbl_seg_status.setText("分析中" if analyzing else ("就绪" if running else "未运行"))
        self._lbl_det_status.setText("检测中" if analyzing else ("就绪" if running else "未运行"))
        self._lbl_source.setText(source if source else "—")

        self._lbl_session_fps.setText(f"FPS: {fps:.1f}" if running else "FPS: —")
        self._lbl_session_pts.setText(f"点数: {points:,}" if running else "点数: —")
        self._lbl_session_obs.setText(f"障碍物: {obstacles}" if analyzing else "障碍物: —")
        self._lbl_session_mode.setText(f"数据源: {source}" if source else "数据源: —")

    def set_busy(self, busy: bool) -> None:
        # busy 期间只禁用启动按钮，停止按钮保持可用，确保用户随时可以中断
        self._btn_start_rt.setEnabled(not busy)
        self._btn_start_ana.setEnabled(not busy)
