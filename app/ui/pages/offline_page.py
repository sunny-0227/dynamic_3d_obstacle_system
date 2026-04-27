from __future__ import annotations

"""
离线点云分析页面

包含：
  - 数据源选择（单文件 / nuScenes mini）
  - 单文件点云选择与加载
  - nuScenes 根目录、连接、场景/帧导航
  - 执行检测、分割、融合显示、一键分析
  - 当前状态信息展示（路径、帧号、点云大小、检测/分割结果）
"""

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class OfflinePage(QWidget):
    """离线点云分析页面，所有信号在此定义，由 MainWindow 连接到 Controller。"""

    # ── 数据源 ──────────────────────────────────────────────────
    sig_data_source_changed  = pyqtSignal(str)   # "single_file" | "nuscenes"

    # ── 单文件 ───────────────────────────────────────────────────
    sig_select_file          = pyqtSignal()
    sig_load_pointcloud      = pyqtSignal()

    # ── nuScenes ─────────────────────────────────────────────────
    sig_select_nusc_root     = pyqtSignal()
    sig_connect_nusc         = pyqtSignal()
    sig_nav_changed          = pyqtSignal(str)
    sig_scene_changed        = pyqtSignal()
    sig_prev_frame           = pyqtSignal()
    sig_next_frame           = pyqtSignal()
    sig_load_current_frame   = pyqtSignal()

    # ── 算法操作 ─────────────────────────────────────────────────
    sig_run_detect           = pyqtSignal()
    sig_run_segment          = pyqtSignal()
    sig_show_fusion          = pyqtSignal()
    sig_run_full             = pyqtSignal()
    sig_clear_results        = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._block_source_emit = False
        self._build_ui()
        self._connect_internal()

    # ------------------------------------------------------------------
    # 构建 UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        # 页面标题
        title = QLabel("离线点云分析")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        # 分割线
        root.addWidget(self._make_hline())

        # 上半部分：数据源 + 操作，左右并排
        content = QHBoxLayout()
        content.setSpacing(16)
        content.addLayout(self._build_left_col(), stretch=1)
        content.addLayout(self._build_right_col(), stretch=1)
        root.addLayout(content)

        # 底部状态信息
        root.addWidget(self._make_hline())
        root.addWidget(self._build_status_panel())
        root.addStretch(1)

    def _build_left_col(self) -> QVBoxLayout:
        """左列：数据源选择 + 单文件/nuScenes 控件。"""
        col = QVBoxLayout()
        col.setSpacing(12)

        # ── 数据源选择 ───────────────────────────────────────────
        src_group = QGroupBox("数据源")
        src_lay = QVBoxLayout(src_group)
        src_lay.setSpacing(8)

        self._radio_single = QRadioButton("单文件点云（.bin / .pcd）")
        self._radio_nusc   = QRadioButton("nuScenes mini 数据集")
        self._btn_grp_src  = QButtonGroup(self)
        self._btn_grp_src.addButton(self._radio_single, 0)
        self._btn_grp_src.addButton(self._radio_nusc,   1)
        self._radio_single.setChecked(True)

        src_lay.addWidget(self._radio_single)
        src_lay.addWidget(self._radio_nusc)
        col.addWidget(src_group)

        # ── 单文件 ────────────────────────────────────────────────
        self._single_group = QGroupBox("单文件点云")
        sg_lay = QVBoxLayout(self._single_group)
        sg_lay.setSpacing(8)

        row_file = QHBoxLayout()
        self._btn_pick_file = QPushButton("选择文件")
        self._lbl_file_path = QLineEdit()
        self._lbl_file_path.setReadOnly(True)
        self._lbl_file_path.setPlaceholderText("未选择文件")
        row_file.addWidget(self._btn_pick_file)
        row_file.addWidget(self._lbl_file_path, stretch=1)
        sg_lay.addLayout(row_file)

        self._btn_load_file = QPushButton("加载点云")
        self._btn_load_file.setEnabled(False)
        self._btn_load_file.setMinimumHeight(34)
        sg_lay.addWidget(self._btn_load_file)
        col.addWidget(self._single_group)

        # ── nuScenes ─────────────────────────────────────────────
        self._nusc_group = QGroupBox("nuScenes mini")
        nlay = QVBoxLayout(self._nusc_group)
        nlay.setSpacing(8)

        row_root = QHBoxLayout()
        self._btn_nusc_root = QPushButton("选择数据集根目录")
        self._path_nusc = QLineEdit()
        self._path_nusc.setReadOnly(True)
        self._path_nusc.setPlaceholderText("未选择")
        self._path_nusc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_root.addWidget(self._btn_nusc_root)
        row_root.addWidget(self._path_nusc, stretch=1)
        nlay.addLayout(row_root)

        self._btn_nusc_connect = QPushButton("加载数据集")
        self._btn_nusc_connect.setEnabled(False)
        self._btn_nusc_connect.setMinimumHeight(34)
        nlay.addWidget(self._btn_nusc_connect)

        row_nav = QHBoxLayout()
        row_nav.addWidget(QLabel("导航方式:"))
        self._combo_mode = QComboBox()
        self._combo_mode.addItem("全数据集（sample 表顺序）", "global")
        self._combo_mode.addItem("按场景时序链", "scene")
        row_nav.addWidget(self._combo_mode, stretch=1)
        nlay.addLayout(row_nav)

        row_scene = QHBoxLayout()
        row_scene.addWidget(QLabel("场景:"))
        self._combo_scene = QComboBox()
        self._combo_scene.setEnabled(False)
        row_scene.addWidget(self._combo_scene, stretch=1)
        nlay.addLayout(row_scene)

        row_frame = QHBoxLayout()
        row_frame.addWidget(QLabel("帧索引:"))
        self._spin_frame = QSpinBox()
        self._spin_frame.setMinimum(0)
        self._spin_frame.setMaximum(0)
        self._spin_frame.setEnabled(False)
        row_frame.addWidget(self._spin_frame)
        self._btn_prev = QPushButton("◀")
        self._btn_next = QPushButton("▶")
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._btn_prev.setFixedWidth(40)
        self._btn_next.setFixedWidth(40)
        row_frame.addWidget(self._btn_prev)
        row_frame.addWidget(self._btn_next)
        nlay.addLayout(row_frame)

        self._btn_load_frame = QPushButton("加载当前帧点云")
        self._btn_load_frame.setEnabled(False)
        self._btn_load_frame.setMinimumHeight(34)
        nlay.addWidget(self._btn_load_frame)

        self._lbl_nusc_meta = QLabel("请先选择根目录并加载数据集")
        self._lbl_nusc_meta.setObjectName("metaLabel")
        self._lbl_nusc_meta.setWordWrap(True)
        nlay.addWidget(self._lbl_nusc_meta)

        col.addWidget(self._nusc_group)
        self._nusc_group.setEnabled(False)  # 默认禁用

        return col

    def _build_right_col(self) -> QVBoxLayout:
        """右列：算法操作区。"""
        col = QVBoxLayout()
        col.setSpacing(12)

        alg_group = QGroupBox("算法操作")
        alg_lay = QVBoxLayout(alg_group)
        alg_lay.setSpacing(10)

        hint = QLabel("请先加载点云后再执行以下算法")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        alg_lay.addWidget(hint)

        self._btn_detect = QPushButton("执行 OpenPCDet 检测")
        self._btn_segment = QPushButton("执行语义分割")
        self._btn_fusion = QPushButton("融合显示结果")
        for b in (self._btn_detect, self._btn_segment, self._btn_fusion):
            b.setMinimumHeight(38)
            b.setEnabled(False)
        alg_lay.addWidget(self._btn_detect)
        alg_lay.addWidget(self._btn_segment)
        alg_lay.addWidget(self._btn_fusion)

        alg_lay.addWidget(self._make_hline())

        self._btn_full = QPushButton("一键分析（检测 + 分割 + 融合）")
        self._btn_full.setObjectName("btnOneClick")
        self._btn_full.setMinimumHeight(48)
        self._btn_full.setEnabled(False)
        alg_lay.addWidget(self._btn_full)

        self._btn_clear = QPushButton("清空结果")
        self._btn_clear.setMinimumHeight(34)
        alg_lay.addWidget(self._btn_clear)

        col.addWidget(alg_group)
        col.addStretch(1)
        return col

    def _build_status_panel(self) -> QWidget:
        """底部状态信息面板。"""
        panel = QGroupBox("当前状态")
        lay = QVBoxLayout(panel)
        lay.setSpacing(4)

        self._lbl_file      = QLabel("点云文件：—")
        self._lbl_points    = QLabel("点云大小：—")
        self._lbl_det_res   = QLabel("检测结果：—")
        self._lbl_seg_res   = QLabel("分割结果：—")

        for lbl in (self._lbl_file, self._lbl_points, self._lbl_det_res, self._lbl_seg_res):
            lbl.setObjectName("statusLabel")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)

        return panel

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
        self._btn_grp_src.buttonClicked.connect(self._on_src_changed)

        self._btn_pick_file.clicked.connect(self.sig_select_file.emit)
        self._btn_load_file.clicked.connect(self.sig_load_pointcloud.emit)

        self._btn_nusc_root.clicked.connect(self.sig_select_nusc_root.emit)
        self._btn_nusc_connect.clicked.connect(self.sig_connect_nusc.emit)
        self._combo_mode.currentIndexChanged.connect(
            lambda _: self.sig_nav_changed.emit(self.navigation_mode())
        )
        self._combo_scene.currentIndexChanged.connect(self.sig_scene_changed.emit)
        self._btn_prev.clicked.connect(self.sig_prev_frame.emit)
        self._btn_next.clicked.connect(self.sig_next_frame.emit)
        self._btn_load_frame.clicked.connect(self.sig_load_current_frame.emit)

        self._btn_detect.clicked.connect(self.sig_run_detect.emit)
        self._btn_segment.clicked.connect(self.sig_run_segment.emit)
        self._btn_fusion.clicked.connect(self.sig_show_fusion.emit)
        self._btn_full.clicked.connect(self.sig_run_full.emit)
        self._btn_clear.clicked.connect(self.sig_clear_results.emit)

    def _on_src_changed(self) -> None:
        if self._block_source_emit:
            return
        src = "nuscenes" if self._radio_nusc.isChecked() else "single_file"
        self._apply_source_lock(src)
        self.sig_data_source_changed.emit(src)

    def _apply_source_lock(self, src: str) -> None:
        is_nusc = src == "nuscenes"
        self._nusc_group.setEnabled(is_nusc)
        self._single_group.setEnabled(not is_nusc)

    # ------------------------------------------------------------------
    # 公开接口（供 MainWindow 调用）
    # ------------------------------------------------------------------

    def ui_data_source(self) -> str:
        return "nuscenes" if self._radio_nusc.isChecked() else "single_file"

    def sync_data_source_radio(self, src: str) -> None:
        self._block_source_emit = True
        if src == "nuscenes":
            self._radio_nusc.setChecked(True)
        else:
            self._radio_single.setChecked(True)
        self._apply_source_lock(src)
        self._block_source_emit = False

    def navigation_mode(self) -> str:
        return self._combo_mode.currentData() or "global"

    def current_scene_token(self) -> str:
        return self._combo_scene.currentData() or ""

    def frame_index(self) -> int:
        return self._spin_frame.value()

    def set_frame_index(self, idx: int) -> None:
        self._spin_frame.setValue(idx)

    def set_selected_file(self, p: Optional[Path]) -> None:
        self._lbl_file_path.setText(str(p) if p else "")
        self._btn_load_file.setEnabled(p is not None)

    def set_nusc_root(self, p: Optional[Path]) -> None:
        self._path_nusc.setText(str(p) if p else "")
        self._btn_nusc_connect.setEnabled(p is not None)

    def set_nusc_meta_text(self, text: str) -> None:
        self._lbl_nusc_meta.setText(text)

    def set_scene_list(self, items: list[tuple[str, str]]) -> None:
        self._combo_scene.blockSignals(True)
        self._combo_scene.clear()
        for name, token in items:
            self._combo_scene.addItem(name, token)
        self._combo_scene.blockSignals(False)

    def set_nusc_nav_enabled(self, enabled: bool, frame_count: int = 0) -> None:
        self._combo_scene.setEnabled(enabled and self.navigation_mode() == "scene")
        self._spin_frame.setEnabled(enabled)
        self._spin_frame.setMaximum(max(0, frame_count - 1))
        for b in (self._btn_prev, self._btn_next, self._btn_load_frame):
            b.setEnabled(enabled)

    def set_action_buttons_state(
        self,
        has_nonempty_pcd: bool,
        has_results: bool,
        allow_run_full_autoload: bool,
        defense_strict_pipeline: bool = True,
    ) -> None:
        self._btn_detect.setEnabled(has_nonempty_pcd)
        self._btn_segment.setEnabled(has_nonempty_pcd)
        self._btn_fusion.setEnabled(has_nonempty_pcd and has_results)
        # 严格流程下必须先有点云；非严格模式下允许自动加载后执行
        can_full = has_nonempty_pcd or (allow_run_full_autoload and not defense_strict_pipeline)
        self._btn_full.setEnabled(can_full)

    def set_busy(self, busy: bool) -> None:
        for w in (
            self._btn_detect, self._btn_segment, self._btn_fusion,
            self._btn_full, self._btn_load_file, self._btn_load_frame,
            self._btn_nusc_connect,
        ):
            w.setEnabled(not busy)

    def update_status(
        self,
        file_path: Optional[Path],
        n_points: int,
        n_detections: int,
        n_seg_classes: int,
    ) -> None:
        self._lbl_file.setText(
            f"点云文件：{file_path.name if file_path else '—'}"
        )
        self._lbl_points.setText(
            f"点云大小：{n_points:,} 点" if n_points > 0 else "点云大小：未载入"
        )
        self._lbl_det_res.setText(
            f"检测结果：{n_detections} 个目标框" if n_detections >= 0 else "检测结果：—"
        )
        self._lbl_seg_res.setText(
            f"分割结果：{n_seg_classes} 类" if n_seg_classes >= 0 else "分割结果：—"
        )
