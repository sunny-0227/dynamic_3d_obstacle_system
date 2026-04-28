from __future__ import annotations

"""
离线点云分析页面

包含：
  - 数据源选择（单文件 / nuScenes mini）
  - 单文件点云选择与加载
  - nuScenes 根目录、连接、场景/帧导航
  - 执行检测、分割、融合显示、一键分析
  - 当前状态信息展示（路径、帧号、点云大小、检测/分割结果）

布局优化记录：
  - 页面外边距：28 24（上下）→ 24 28（左右）
  - 各 GroupBox 间距：spacing=18
  - GroupBox 内部 spacing=10，内边距 12px
  - 输入框最小高度 32px，最小宽度 200px
  - 算法按钮高度提升：38→42，一键分析：48→54
  - 左右列比例 stretch=1:1，均等分配
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
        # 【间距优化】外边距：顶部/底部 24px，左/右 28px；控件间距 18px
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(18)

        # 页面标题
        title = QLabel("离线点云分析")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        root.addWidget(self._make_hline())

        # 上半部分：数据源 + 操作，左右均等
        content = QHBoxLayout()
        content.setSpacing(24)           # 【间距优化】左右列间距 24px
        left_col = self._build_left_col()
        right_col = self._build_right_col()
        content.addLayout(left_col, stretch=1)    # 【拉伸】左列 stretch=1
        content.addLayout(right_col, stretch=1)   # 【拉伸】右列 stretch=1
        root.addLayout(content, stretch=1)         # 让内容区撑满剩余高度

        # 底部状态信息
        root.addWidget(self._make_hline())
        root.addWidget(self._build_status_panel())

    def _build_left_col(self) -> QVBoxLayout:
        """左列：数据源选择 + 单文件/nuScenes 控件。"""
        col = QVBoxLayout()
        col.setSpacing(18)               # 【间距优化】GroupBox 间距 18px

        # ── 数据源选择 ───────────────────────────────────────────
        src_group = QGroupBox("数据源")
        src_lay = QVBoxLayout(src_group)
        src_lay.setSpacing(10)           # 【间距优化】内部行间距 10px
        src_lay.setContentsMargins(14, 18, 14, 14)  # 【间距优化】组内边距

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
        sg_lay.setSpacing(10)
        sg_lay.setContentsMargins(14, 18, 14, 14)

        row_file = QHBoxLayout()
        row_file.setSpacing(10)
        self._btn_pick_file = QPushButton("选择文件")
        self._btn_pick_file.setMinimumHeight(34)  # 【间距优化】按钮最小高度 34px
        self._btn_pick_file.setFixedWidth(90)
        self._lbl_file_path = QLineEdit()
        self._lbl_file_path.setReadOnly(True)
        self._lbl_file_path.setPlaceholderText("未选择文件")
        self._lbl_file_path.setMinimumWidth(200)  # 【最小宽度】防止被压缩
        self._lbl_file_path.setMinimumHeight(34)
        row_file.addWidget(self._btn_pick_file)
        row_file.addWidget(self._lbl_file_path, stretch=1)
        sg_lay.addLayout(row_file)

        self._btn_load_file = QPushButton("加载点云")
        self._btn_load_file.setEnabled(False)
        self._btn_load_file.setMinimumHeight(38)  # 【间距优化】加载按钮高 38px
        sg_lay.addWidget(self._btn_load_file)
        col.addWidget(self._single_group)

        # ── nuScenes ─────────────────────────────────────────────
        self._nusc_group = QGroupBox("nuScenes mini")
        # Minimum 策略：GroupBox 按内容自动撑开高度，不压缩子控件
        self._nusc_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        nlay = QVBoxLayout(self._nusc_group)
        nlay.setSpacing(12)
        nlay.setContentsMargins(12, 16, 12, 12)

        # 行1：根目录选择
        row_root = QHBoxLayout()
        row_root.setSpacing(10)
        self._btn_nusc_root = QPushButton("选择根目录")
        self._btn_nusc_root.setMinimumHeight(34)
        self._btn_nusc_root.setFixedWidth(100)
        self._path_nusc = QLineEdit()
        self._path_nusc.setReadOnly(True)
        self._path_nusc.setPlaceholderText("未选择数据集根目录")
        self._path_nusc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._path_nusc.setMinimumWidth(200)
        self._path_nusc.setMinimumHeight(34)
        row_root.addWidget(self._btn_nusc_root)
        row_root.addWidget(self._path_nusc, stretch=1)
        nlay.addLayout(row_root)

        # 行2：加载数据集
        self._btn_nusc_connect = QPushButton("加载数据集")
        self._btn_nusc_connect.setEnabled(False)
        self._btn_nusc_connect.setMinimumHeight(36)
        nlay.addWidget(self._btn_nusc_connect)

        # 行3：导航方式
        row_nav = QHBoxLayout()
        row_nav.setSpacing(10)
        lbl_nav = QLabel("导航方式:")
        lbl_nav.setMinimumWidth(72)
        lbl_nav.setMinimumHeight(32)
        self._combo_mode = QComboBox()
        self._combo_mode.addItem("全数据集（sample 表顺序）", "global")
        self._combo_mode.addItem("按场景时序链", "scene")
        self._combo_mode.setMinimumHeight(32)
        row_nav.addWidget(lbl_nav)
        row_nav.addWidget(self._combo_mode, stretch=1)
        nlay.addLayout(row_nav)

        # 行4：场景选择
        row_scene = QHBoxLayout()
        row_scene.setSpacing(10)
        lbl_scene = QLabel("场景:")
        lbl_scene.setMinimumWidth(72)
        lbl_scene.setMinimumHeight(32)
        self._combo_scene = QComboBox()
        self._combo_scene.setEnabled(False)
        self._combo_scene.setMinimumHeight(32)
        row_scene.addWidget(lbl_scene)
        row_scene.addWidget(self._combo_scene, stretch=1)
        nlay.addLayout(row_scene)

        # 行5：帧索引 + 翻页按钮（统一 minimumHeight，避免高度不一致导致重叠）
        row_frame = QHBoxLayout()
        row_frame.setSpacing(10)
        lbl_frame = QLabel("帧索引:")
        lbl_frame.setMinimumWidth(72)
        lbl_frame.setMinimumHeight(34)
        self._spin_frame = QSpinBox()
        self._spin_frame.setMinimum(0)
        self._spin_frame.setMaximum(0)
        self._spin_frame.setEnabled(False)
        self._spin_frame.setMinimumHeight(34)
        self._spin_frame.setMaximumWidth(100)
        self._btn_prev = QPushButton("◀")
        self._btn_next = QPushButton("▶")
        self._btn_prev.setEnabled(False)
        self._btn_next.setEnabled(False)
        self._btn_prev.setFixedWidth(44)
        self._btn_prev.setMinimumHeight(34)
        self._btn_next.setFixedWidth(44)
        self._btn_next.setMinimumHeight(34)
        row_frame.addWidget(lbl_frame)
        row_frame.addWidget(self._spin_frame)
        row_frame.addStretch(1)
        row_frame.addWidget(self._btn_prev)
        row_frame.addWidget(self._btn_next)
        nlay.addLayout(row_frame)

        # 行6：加载当前帧
        self._btn_load_frame = QPushButton("加载当前帧点云")
        self._btn_load_frame.setEnabled(False)
        self._btn_load_frame.setMinimumHeight(36)
        nlay.addWidget(self._btn_load_frame)

        # 元信息提示行
        self._lbl_nusc_meta = QLabel("请先选择根目录并加载数据集")
        self._lbl_nusc_meta.setObjectName("metaLabel")
        self._lbl_nusc_meta.setWordWrap(True)
        self._lbl_nusc_meta.setMinimumHeight(32)
        nlay.addWidget(self._lbl_nusc_meta)

        col.addWidget(self._nusc_group)
        self._nusc_group.setEnabled(False)

        # 底部 stretch：允许左列在窗口高度充足时自然撑开，高度不足时不压缩 nusc_group
        col.addStretch(1)
        return col

    def _build_right_col(self) -> QVBoxLayout:
        """右列：算法操作区。"""
        col = QVBoxLayout()
        col.setSpacing(18)

        alg_group = QGroupBox("算法操作")
        alg_lay = QVBoxLayout(alg_group)
        alg_lay.setSpacing(14)            # 【间距优化】按钮行间距 14px
        alg_lay.setContentsMargins(16, 20, 16, 16)

        hint = QLabel("请先加载点云后再执行以下算法")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        alg_lay.addWidget(hint)

        alg_lay.addSpacing(4)             # 【间距优化】提示文字与按钮额外间距

        self._btn_detect  = QPushButton("执行 OpenPCDet 检测")
        self._btn_segment = QPushButton("执行语义分割")
        self._btn_fusion  = QPushButton("融合显示结果")
        for b in (self._btn_detect, self._btn_segment, self._btn_fusion):
            b.setMinimumHeight(42)        # 【间距优化】操作按钮高 42px
            b.setEnabled(False)
        alg_lay.addWidget(self._btn_detect)
        alg_lay.addWidget(self._btn_segment)
        alg_lay.addWidget(self._btn_fusion)

        alg_lay.addSpacing(6)
        alg_lay.addWidget(self._make_hline())
        alg_lay.addSpacing(6)

        self._btn_full = QPushButton("一键分析（检测 + 分割 + 融合）")
        self._btn_full.setObjectName("btnOneClick")
        self._btn_full.setMinimumHeight(54)   # 【间距优化】主操作按钮高 54px
        self._btn_full.setEnabled(False)
        alg_lay.addWidget(self._btn_full)

        alg_lay.addSpacing(4)

        self._btn_clear = QPushButton("清空结果")
        self._btn_clear.setMinimumHeight(36)
        alg_lay.addWidget(self._btn_clear)

        col.addWidget(alg_group)
        col.addStretch(1)                 # 【拉伸】右列底部留白
        return col

    def _build_status_panel(self) -> QWidget:
        """底部状态信息面板。"""
        panel = QGroupBox("当前状态")
        lay = QHBoxLayout(panel)          # 【间距优化】改为横向排列，更紧凑
        lay.setSpacing(32)
        lay.setContentsMargins(16, 14, 16, 14)

        self._lbl_file   = QLabel("点云文件：—")
        self._lbl_points = QLabel("点云大小：—")
        self._lbl_det_res = QLabel("检测结果：—")
        self._lbl_seg_res = QLabel("分割结果：—")

        for lbl in (self._lbl_file, self._lbl_points, self._lbl_det_res, self._lbl_seg_res):
            lbl.setObjectName("statusLabel")
            lbl.setWordWrap(False)
            lay.addWidget(lbl, stretch=1)  # 【拉伸】四列均等拉伸

        return panel

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
    # 公开接口（不改变）
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
