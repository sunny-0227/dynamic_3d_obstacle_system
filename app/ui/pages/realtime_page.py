from __future__ import annotations

"""
实时相机感知页面（性能优化版）

性能优化新增显示项：
  - 相机 FPS（采集帧率）
  - 处理 FPS（算法帧率）
  - 原始点数（采集后未下采样）
  - 下采样后点数（送入算法）
  - 障碍物数量
  - 当前帧处理耗时

布局：左右两列，左列为控制区，右列为仪表盘。
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

    sig_camera_source_changed   = pyqtSignal(str)   # "realsense" | "mock"
    sig_select_realtime_dir     = pyqtSignal()
    sig_start_realtime          = pyqtSignal()
    sig_stop_realtime           = pyqtSignal()
    sig_start_realtime_analysis = pyqtSignal()
    sig_stop_realtime_analysis  = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._connect_internal()

    # ------------------------------------------------------------------
    # 构建 UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        title = QLabel("实时相机感知")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(self._make_hline())

        body = QHBoxLayout()
        body.setSpacing(24)
        body.addLayout(self._build_left_col(),  stretch=1)
        body.addLayout(self._build_right_col(), stretch=1)
        root.addLayout(body, stretch=1)

        root.addWidget(self._make_hline())
        root.addWidget(self._build_status_panel())

    # ── 左列：控制区 ──────────────────────────────────────────────────

    def _build_left_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(18)

        # 数据源选择
        src_group = QGroupBox("数据源选择")
        sg = QVBoxLayout(src_group)
        sg.setSpacing(10)
        sg.setContentsMargins(14, 18, 14, 14)

        self._radio_realsense = QRadioButton("Intel RealSense 深度相机")
        self._radio_mock      = QRadioButton("Mock 点云流（本地文件循环）")
        self._btn_grp_cam     = QButtonGroup(self)
        self._btn_grp_cam.addButton(self._radio_realsense, 0)
        self._btn_grp_cam.addButton(self._radio_mock,      1)
        self._radio_realsense.setChecked(True)
        sg.addWidget(self._radio_realsense)
        sg.addWidget(self._radio_mock)

        hint_src = QLabel("RealSense 直接连接硬件；Mock 从本地目录循环读取点云文件。")
        hint_src.setObjectName("hintLabel")
        hint_src.setWordWrap(True)
        sg.addWidget(hint_src)
        col.addWidget(src_group)

        # Mock 目录
        self._mock_group = QGroupBox("Mock 点云流目录")
        mg = QHBoxLayout(self._mock_group)
        mg.setSpacing(10)
        mg.setContentsMargins(14, 18, 14, 14)
        self._btn_pick_dir = QPushButton("选择目录")
        self._btn_pick_dir.setMinimumHeight(34)
        self._btn_pick_dir.setFixedWidth(90)
        self._lbl_rt_dir = QLineEdit()
        self._lbl_rt_dir.setReadOnly(True)
        self._lbl_rt_dir.setPlaceholderText("未选择目录")
        self._lbl_rt_dir.setMinimumWidth(200)
        self._lbl_rt_dir.setMinimumHeight(34)
        mg.addWidget(self._btn_pick_dir)
        mg.addWidget(self._lbl_rt_dir, stretch=1)
        self._mock_group.setEnabled(False)
        col.addWidget(self._mock_group)

        # 模式控制按钮
        ctrl_group = QGroupBox("实时模式控制")
        cl = QVBoxLayout(ctrl_group)
        cl.setSpacing(12)
        cl.setContentsMargins(14, 18, 14, 16)

        self._btn_start_rt = QPushButton("▶  启动实时模式")
        self._btn_stop_rt  = QPushButton("■  停止实时模式")
        self._btn_start_rt.setMinimumHeight(46)
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
        col.addStretch(1)
        return col

    # ── 右列：仪表盘 ─────────────────────────────────────────────────

    def _build_right_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(18)

        # 主仪表盘（新增性能指标）
        dash_group = QGroupBox("实时状态仪表盘")
        dl = QVBoxLayout(dash_group)
        dl.setSpacing(12)
        dl.setContentsMargins(14, 20, 14, 16)

        self._lbl_cam_status  = self._make_status_row(dl, "相机连接状态",    "未连接")
        self._lbl_cam_fps     = self._make_status_row(dl, "相机采集 FPS",    "—")
        self._lbl_proc_fps    = self._make_status_row(dl, "算法处理 FPS",    "—")
        self._lbl_raw_points  = self._make_status_row(dl, "原始点数",        "—")
        self._lbl_proc_points = self._make_status_row(dl, "下采样后点数",    "—")
        self._lbl_obstacles   = self._make_status_row(dl, "当前障碍物数",    "—")
        self._lbl_proc_time   = self._make_status_row(dl, "处理耗时 (ms)",   "—")
        self._lbl_seg_status  = self._make_status_row(dl, "实时分割状态",    "未运行")
        self._lbl_det_status  = self._make_status_row(dl, "实时检测框状态",  "未运行")
        self._lbl_source      = self._make_status_row(dl, "数据来源",        "—")

        col.addWidget(dash_group)

        # 操作说明
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
            "提示：调低分辨率或增大 voxel_size 可显著提升 FPS",
        ]
        for t in tips:
            lbl = QLabel(t)
            lbl.setObjectName("hintLabel")
            lbl.setMinimumHeight(22)
            tl.addWidget(lbl)

        col.addWidget(tip_group)
        col.addStretch(1)
        return col

    # ── 底部状态栏 ───────────────────────────────────────────────────

    def _build_status_panel(self) -> QWidget:
        panel = QGroupBox("实时会话信息")
        lay = QHBoxLayout(panel)
        lay.setSpacing(16)
        lay.setContentsMargins(16, 14, 16, 14)

        self._lbl_session_cam_fps  = QLabel("相机FPS: —")
        self._lbl_session_proc_fps = QLabel("处理FPS: —")
        self._lbl_session_pts      = QLabel("原始点数: —")
        self._lbl_session_dpts     = QLabel("处理点数: —")
        self._lbl_session_obs      = QLabel("障碍物: —")
        self._lbl_session_time     = QLabel("耗时: — ms")

        for lbl in (
            self._lbl_session_cam_fps, self._lbl_session_proc_fps,
            self._lbl_session_pts, self._lbl_session_dpts,
            self._lbl_session_obs, self._lbl_session_time,
        ):
            lbl.setObjectName("statusLabel")
            lbl.setMinimumHeight(22)
            lay.addWidget(lbl, stretch=1)

        return panel

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _make_status_row(parent_lay: QVBoxLayout, label: str, default: str) -> QLabel:
        """创建一行 key: value 状态显示，key 固定宽度保证对齐。"""
        row = QHBoxLayout()
        row.setSpacing(12)
        key = QLabel(f"{label}：")
        key.setFixedWidth(150)     # 稍宽以容纳"下采样后点数"等较长文字
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
    # 公开接口
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
        camera_fps: float = 0.0,
        process_fps: float = 0.0,
        raw_points: int = 0,
        proc_points: int = 0,
        proc_elapsed_ms: float = 0.0,
    ) -> None:
        """
        更新仪表盘所有字段。

        参数（新增）：
          camera_fps      — 相机采集帧率
          process_fps     — 算法处理帧率
          raw_points      — 相机原始点数
          proc_points     — 下采样后算法处理点数
          proc_elapsed_ms — 本帧算法耗时（毫秒）
        """
        cam_txt = "已连接 / 运行中" if running else "未连接"
        self._lbl_cam_status.setText(cam_txt)

        # 相机 FPS（以相机采集 FPS 为准；兼容旧版 fps 参数）
        display_cam_fps = camera_fps if camera_fps > 0 else fps
        self._lbl_cam_fps.setText(f"{display_cam_fps:.1f}" if running else "—")
        self._lbl_proc_fps.setText(f"{process_fps:.1f}" if (running and analyzing and process_fps > 0) else "—")

        # 点数显示
        disp_raw = raw_points if raw_points > 0 else points
        self._lbl_raw_points.setText(f"{disp_raw:,}" if running else "—")
        self._lbl_proc_points.setText(
            f"{proc_points:,}" if (running and analyzing and proc_points > 0) else "—"
        )

        self._lbl_obstacles.setText(f"{obstacles}" if analyzing else "—")
        self._lbl_proc_time.setText(
            f"{proc_elapsed_ms:.0f}" if (running and analyzing and proc_elapsed_ms > 0) else "—"
        )
        self._lbl_seg_status.setText("分析中" if analyzing else ("就绪" if running else "未运行"))
        self._lbl_det_status.setText("检测中" if analyzing else ("就绪" if running else "未运行"))
        self._lbl_source.setText(source if source else "—")

        # 底部汇总栏
        self._lbl_session_cam_fps.setText(
            f"相机FPS: {display_cam_fps:.1f}" if running else "相机FPS: —"
        )
        self._lbl_session_proc_fps.setText(
            f"处理FPS: {process_fps:.1f}" if (running and analyzing and process_fps > 0) else "处理FPS: —"
        )
        self._lbl_session_pts.setText(
            f"原始点数: {disp_raw:,}" if running else "原始点数: —"
        )
        self._lbl_session_dpts.setText(
            f"处理点数: {proc_points:,}" if (running and analyzing and proc_points > 0) else "处理点数: —"
        )
        self._lbl_session_obs.setText(
            f"障碍物: {obstacles}" if analyzing else "障碍物: —"
        )
        self._lbl_session_time.setText(
            f"耗时: {proc_elapsed_ms:.0f} ms" if (running and analyzing and proc_elapsed_ms > 0) else "耗时: — ms"
        )

    def set_busy(self, busy: bool) -> None:
        # busy 期间只禁用启动按钮，停止按钮保持可用，确保用户随时可以中断
        self._btn_start_rt.setEnabled(not busy)
        self._btn_start_ana.setEnabled(not busy)
