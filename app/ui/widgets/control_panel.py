"""
左侧控制面板（科研/毕设主界面）

模块顺序：数据源模式 → nuScenes mini → 单文件点云 → 执行处理。
通过单选框区分数据源，避免两种模式控件同时可操作造成混淆。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
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


class ControlPanel(QWidget):
    # 数据源（界面层首选，与控制器 workflow 同步）
    sig_data_source_changed = pyqtSignal(str)  # "single_file" | "nuscenes"

    sig_select_file = pyqtSignal()
    sig_load_pointcloud = pyqtSignal()

    sig_select_nusc_root = pyqtSignal()
    sig_connect_nusc = pyqtSignal()
    sig_nav_changed = pyqtSignal(str)
    sig_scene_changed = pyqtSignal()
    sig_prev_frame = pyqtSignal()
    sig_next_frame = pyqtSignal()
    sig_load_current_frame = pyqtSignal()

    sig_run_detect = pyqtSignal()
    sig_run_segment = pyqtSignal()
    sig_show_fusion = pyqtSignal()
    sig_run_full = pyqtSignal()
    sig_clear_results = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_cache: dict[int, bool] = {}
        self._block_source_emit = False
        self._fixed_width = 420
        self.setFixedWidth(self._fixed_width)

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(0, 0, 8, 0)

        # ---------- 1) 数据源模式 ----------
        mode_group = QGroupBox("① 数据源模式")
        mode_lay = QVBoxLayout(mode_group)
        mode_lay.setSpacing(8)
        self._radio_single = QRadioButton("单文件点云（.bin / .pcd）")
        self._radio_nusc = QRadioButton("nuScenes mini 数据集")
        self._btn_group_source = QButtonGroup(self)
        self._btn_group_source.addButton(self._radio_single, 0)
        self._btn_group_source.addButton(self._radio_nusc, 1)
        self._radio_single.setChecked(True)
        mode_lay.addWidget(self._radio_single)
        mode_lay.addWidget(self._radio_nusc)
        hint_mode = QLabel("请先选择一种数据源；未选中的模块将禁用，避免误操作。")
        hint_mode.setWordWrap(True)
        hint_mode.setStyleSheet("color: #6c7086; font-size: 11px;")
        mode_lay.addWidget(hint_mode)
        root.addWidget(mode_group)

        # ---------- 2) nuScenes ----------
        self._nusc_group = QGroupBox("② nuScenes mini")
        nusc_layout = QVBoxLayout(self._nusc_group)
        nusc_layout.setSpacing(6)

        row_root = QHBoxLayout()
        self._btn_nusc_root = QPushButton("选择数据集根目录")
        self._path_nusc = QLineEdit()
        self._path_nusc.setReadOnly(True)
        self._path_nusc.setPlaceholderText("未选择")
        self._path_nusc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._path_nusc.setMinimumHeight(30)
        row_root.addWidget(self._btn_nusc_root)
        row_root.addWidget(self._path_nusc, stretch=1)
        nusc_layout.addLayout(row_root)

        row_connect = QHBoxLayout()
        self._btn_nusc_connect = QPushButton("加载数据集")
        self._btn_nusc_connect.setEnabled(False)
        self._btn_nusc_connect.setMinimumHeight(34)
        self._btn_nusc_connect.setObjectName("btnLoadDataset")
        row_connect.addWidget(self._btn_nusc_connect)
        nusc_layout.addLayout(row_connect)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("导航方式:"))
        self._combo_mode = QComboBox()
        self._combo_mode.addItem("全数据集（sample 表顺序，非时间序）", "global")
        self._combo_mode.addItem("按场景时序链", "scene")
        self._combo_mode.setCurrentIndex(0)
        row_mode.addWidget(self._combo_mode, stretch=1)
        nusc_layout.addLayout(row_mode)

        row_scene = QHBoxLayout()
        row_scene.addWidget(QLabel("场景:"))
        self._combo_scene = QComboBox()
        self._combo_scene.setEnabled(False)
        row_scene.addWidget(self._combo_scene, stretch=1)
        nusc_layout.addLayout(row_scene)

        row_frame = QHBoxLayout()
        row_frame.addWidget(QLabel("帧索引:"))
        self._spin_frame = QSpinBox()
        self._spin_frame.setMinimum(0)
        self._spin_frame.setMaximum(0)
        self._spin_frame.setEnabled(False)
        row_frame.addWidget(self._spin_frame)
        self._btn_prev = QPushButton("上一帧")
        self._btn_next = QPushButton("下一帧")
        self._btn_load_frame = QPushButton("加载当前帧点云")
        for b in (self._btn_prev, self._btn_next, self._btn_load_frame):
            b.setEnabled(False)
            b.setMinimumHeight(32)
        row_frame.addWidget(self._btn_prev)
        row_frame.addWidget(self._btn_next)
        row_frame.addWidget(self._btn_load_frame)
        nusc_layout.addLayout(row_frame)

        self._label_nusc_meta = QLabel("请先选择根目录并加载数据集")
        self._label_nusc_meta.setWordWrap(True)
        self._label_nusc_meta.setStyleSheet("color: #bac2de; font-size: 11px;")
        nusc_layout.addWidget(self._label_nusc_meta)

        hint_nusc = QLabel(
            "连接后默认「全数据集」导航；可直接调帧并点「加载当前帧点云」。"
        )
        hint_nusc.setWordWrap(True)
        hint_nusc.setStyleSheet("color: #6c7086; font-size: 11px;")
        nusc_layout.addWidget(hint_nusc)
        root.addWidget(self._nusc_group)

        # ---------- 3) 单文件 ----------
        self._file_group = QGroupBox("③ 单文件点云")
        file_layout = QVBoxLayout(self._file_group)
        file_layout.setSpacing(6)
        row_file = QHBoxLayout()
        self._btn_select_file = QPushButton("选择点云文件")
        self._path_file = QLineEdit()
        self._path_file.setReadOnly(True)
        self._path_file.setPlaceholderText("未选择")
        self._path_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._path_file.setMinimumHeight(30)
        row_file.addWidget(self._btn_select_file)
        row_file.addWidget(self._path_file, stretch=1)
        file_layout.addLayout(row_file)
        self._btn_load_file = QPushButton("加载点云")
        self._btn_load_file.setEnabled(False)
        self._btn_load_file.setMinimumHeight(34)
        self._btn_load_file.setObjectName("btnLoadPointcloud")
        file_layout.addWidget(self._btn_load_file)
        root.addWidget(self._file_group)

        # ---------- 4) 执行处理 ----------
        op_group = QGroupBox("④ 执行处理")
        op_layout = QVBoxLayout(op_group)
        op_layout.setSpacing(8)

        row_alg = QHBoxLayout()
        self._btn_detect = QPushButton("执行检测")
        self._btn_segment = QPushButton("执行分割")
        self._btn_fusion = QPushButton("融合显示")
        for b in (self._btn_detect, self._btn_segment, self._btn_fusion):
            b.setMinimumHeight(36)
            b.setEnabled(False)
        row_alg.addWidget(self._btn_detect)
        row_alg.addWidget(self._btn_segment)
        row_alg.addWidget(self._btn_fusion)
        op_layout.addLayout(row_alg)

        self._btn_full = QPushButton("一键分析（检测 + 分割 + 融合）")
        self._btn_full.setObjectName("btnOneClick")
        self._btn_full.setMinimumHeight(44)
        self._btn_full.setEnabled(False)
        op_layout.addWidget(self._btn_full)

        self._btn_clear = QPushButton("清空结果")
        self._btn_clear.setMinimumHeight(34)
        op_layout.addWidget(self._btn_clear)

        root.addWidget(op_group)
        root.addStretch(1)

        # 信号：用 buttonClicked 避免两个 QRadioButton 的 toggled 连续触发两次
        self._btn_group_source.buttonClicked.connect(self._on_source_button_clicked)

        self._btn_select_file.clicked.connect(self.sig_select_file.emit)
        self._btn_load_file.clicked.connect(self.sig_load_pointcloud.emit)

        self._btn_nusc_root.clicked.connect(self.sig_select_nusc_root.emit)
        self._btn_nusc_connect.clicked.connect(self.sig_connect_nusc.emit)
        self._combo_mode.currentIndexChanged.connect(lambda _i: self.sig_nav_changed.emit(self.navigation_mode()))
        self._combo_scene.currentIndexChanged.connect(self.sig_scene_changed.emit)
        self._btn_prev.clicked.connect(self.sig_prev_frame.emit)
        self._btn_next.clicked.connect(self.sig_next_frame.emit)
        self._btn_load_frame.clicked.connect(self.sig_load_current_frame.emit)

        self._btn_detect.clicked.connect(self.sig_run_detect.emit)
        self._btn_segment.clicked.connect(self.sig_run_segment.emit)
        self._btn_fusion.clicked.connect(self.sig_show_fusion.emit)
        self._btn_full.clicked.connect(self.sig_run_full.emit)
        self._btn_clear.clicked.connect(self.sig_clear_results.emit)

        self._busy_widgets = (
            self._radio_single,
            self._radio_nusc,
            self._btn_select_file,
            self._btn_load_file,
            self._btn_nusc_root,
            self._btn_nusc_connect,
            self._combo_mode,
            self._combo_scene,
            self._spin_frame,
            self._btn_prev,
            self._btn_next,
            self._btn_load_frame,
            self._btn_detect,
            self._btn_segment,
            self._btn_fusion,
            self._btn_full,
        )

    def _on_source_button_clicked(self, _btn) -> None:
        if self._block_source_emit:
            return
        if self._radio_nusc.isChecked():
            self.sig_data_source_changed.emit("nuscenes")
        else:
            self.sig_data_source_changed.emit("single_file")

    def ui_data_source(self) -> str:
        return "nuscenes" if self._radio_nusc.isChecked() else "single_file"

    def sync_data_source_radio(self, workflow: str) -> None:
        """根据控制器 workflow 同步单选（不向外发 sig_data_source_changed）。"""
        self._block_source_emit = True
        try:
            if workflow == "nuscenes":
                self._radio_nusc.setChecked(True)
            elif workflow == "single_file":
                self._radio_single.setChecked(True)
            # workflow == none 时保持用户当前选择
        finally:
            self._block_source_emit = False

    def apply_source_module_lock(self, ui_source: str) -> None:
        """仅启用当前数据源对应分组（单选框始终可用）。"""
        nusc_on = ui_source == "nuscenes"
        single_on = ui_source == "single_file"
        self._nusc_group.setEnabled(nusc_on)
        self._file_group.setEnabled(single_on)

    # ----------------------------
    # 状态/数据读写
    # ----------------------------
    def set_selected_file(self, path: Optional[Path]) -> None:
        if path is None:
            self._path_file.clear()
            self._path_file.setPlaceholderText("未选择")
            self._btn_load_file.setEnabled(False)
        else:
            self._path_file.setText(str(path))
            if self._file_group.isEnabled():
                self._btn_load_file.setEnabled(True)

    def set_nusc_root(self, path: Optional[Path]) -> None:
        if path is None:
            self._path_nusc.clear()
            self._path_nusc.setPlaceholderText("未选择")
            self._btn_nusc_connect.setEnabled(False)
        else:
            self._path_nusc.setText(str(path))
            if self._nusc_group.isEnabled():
                self._btn_nusc_connect.setEnabled(True)

    def set_nusc_meta_text(self, text: str) -> None:
        self._label_nusc_meta.setText(str(text))

    def navigation_mode(self) -> str:
        return str(self._combo_mode.currentData())

    def set_scene_list(self, items: list[tuple[str, str]]) -> None:
        self._combo_scene.blockSignals(True)
        self._combo_scene.clear()
        for name, token in items:
            self._combo_scene.addItem(name, token)
        self._combo_scene.blockSignals(False)

    def current_scene_token(self) -> Optional[str]:
        if self._combo_scene.count() <= 0:
            return None
        return str(self._combo_scene.currentData())

    def set_nusc_nav_enabled(self, enabled: bool, frame_count: int = 0) -> None:
        root_set = bool(self._path_nusc.text().strip())
        self._combo_mode.setEnabled((enabled or root_set) and self._nusc_group.isEnabled())

        mode = self.navigation_mode()
        self._combo_scene.setEnabled(enabled and mode == "scene" and self._nusc_group.isEnabled())

        n = int(frame_count)
        nav_ok = enabled and self._nusc_group.isEnabled()
        self._spin_frame.setEnabled(nav_ok and n > 0)
        self._spin_frame.setMaximum(max(0, n - 1))
        if n <= 0:
            self._spin_frame.setValue(0)
        self._btn_prev.setEnabled(nav_ok and n > 0)
        self._btn_next.setEnabled(nav_ok and n > 0)
        self._btn_load_frame.setEnabled(nav_ok and n > 0)

    def frame_index(self) -> int:
        return int(self._spin_frame.value())

    def set_frame_index(self, idx: int) -> None:
        self._spin_frame.setValue(int(idx))

    def set_action_buttons_state(
        self,
        *,
        has_nonempty_pcd: bool,
        has_results: bool,
        allow_run_full_autoload: bool,
        defense_strict_pipeline: bool,
    ) -> None:
        """
        defense_strict_pipeline=True：检测/分割/融合/一键分析均要求已载入非空点云；
        一键分析不再在未加载点云时通过「自动加载」启用（流程约束更严格）。
        """
        can = bool(has_nonempty_pcd)
        self._btn_detect.setEnabled(can)
        self._btn_segment.setEnabled(can)
        self._btn_fusion.setEnabled(can and bool(has_results))
        if defense_strict_pipeline:
            self._btn_full.setEnabled(can)
        else:
            self._btn_full.setEnabled(can or bool(allow_run_full_autoload))

    def set_busy(self, busy: bool) -> None:
        if busy:
            self._enabled_cache = {id(w): bool(w.isEnabled()) for w in self._busy_widgets}
            for w in self._busy_widgets:
                w.setEnabled(False)
        else:
            for w in self._busy_widgets:
                prev = self._enabled_cache.get(id(w), True)
                w.setEnabled(bool(prev))
            self._enabled_cache = {}
