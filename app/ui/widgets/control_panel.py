"""
控制面板组件（里程碑 6）
集成：
  - 单文件点云选择/加载
  - nuScenes mini 根目录选择/加载 + 场景/帧导航
  - 操作按钮：加载点云/检测/分割/融合显示/一键运行/清空结果
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ControlPanel(QWidget):
    # 文件/数据集选择信号
    sig_select_file = pyqtSignal()
    sig_load_pointcloud = pyqtSignal()

    sig_select_nusc_root = pyqtSignal()
    sig_connect_nusc = pyqtSignal()
    sig_nav_changed = pyqtSignal(str)  # "global" / "scene"
    sig_scene_changed = pyqtSignal()
    sig_prev_frame = pyqtSignal()
    sig_next_frame = pyqtSignal()
    sig_load_current_frame = pyqtSignal()

    # 算法/显示
    sig_run_detect = pyqtSignal()
    sig_run_segment = pyqtSignal()
    sig_show_fusion = pyqtSignal()
    sig_run_full = pyqtSignal()
    sig_clear_results = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._enabled_cache: dict[int, bool] = {}
        root = QVBoxLayout(self)
        root.setSpacing(12)

        # ---- 单文件 ----
        file_group = QGroupBox("本地点云文件")
        file_layout = QVBoxLayout(file_group)
        row_file = QHBoxLayout()
        self._btn_select_file = QPushButton("选择点云文件")
        self._label_file = QLabel("未选择")
        self._label_file.setWordWrap(True)
        row_file.addWidget(self._btn_select_file)
        row_file.addWidget(self._label_file, stretch=1)
        file_layout.addLayout(row_file)
        self._btn_load_file = QPushButton("加载点云")
        self._btn_load_file.setEnabled(False)
        file_layout.addWidget(self._btn_load_file)
        root.addWidget(file_group)

        # ---- nuScenes ----
        nusc_group = QGroupBox("nuScenes mini")
        nusc_layout = QVBoxLayout(nusc_group)

        row_root = QHBoxLayout()
        self._btn_nusc_root = QPushButton("选择数据集根目录")
        self._label_nusc_root = QLabel("未选择")
        self._label_nusc_root.setWordWrap(True)
        row_root.addWidget(self._btn_nusc_root)
        row_root.addWidget(self._label_nusc_root, stretch=1)
        nusc_layout.addLayout(row_root)

        row_conn = QHBoxLayout()
        self._btn_nusc_connect = QPushButton("加载数据集")
        self._btn_nusc_connect.setEnabled(False)
        row_conn.addWidget(self._btn_nusc_connect)
        row_conn.addStretch()
        nusc_layout.addLayout(row_conn)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("导航方式:"))
        self._combo_mode = QComboBox()
        self._combo_mode.addItem("全数据集（sample 顺序）", "global")
        self._combo_mode.addItem("按场景时序链", "scene")
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
        row_frame.addWidget(self._btn_prev)
        row_frame.addWidget(self._btn_next)
        row_frame.addWidget(self._btn_load_frame)
        nusc_layout.addLayout(row_frame)

        self._label_nusc_meta = QLabel("请先选择并加载 nuScenes mini 根目录")
        self._label_nusc_meta.setWordWrap(True)
        nusc_layout.addWidget(self._label_nusc_meta)

        root.addWidget(nusc_group)

        # ---- 操作按钮 ----
        op_group = QGroupBox("操作")
        op_layout = QHBoxLayout(op_group)
        op_layout.setSpacing(10)
        self._btn_detect = QPushButton("执行检测")
        self._btn_segment = QPushButton("执行分割")
        self._btn_fusion = QPushButton("融合显示")
        self._btn_full = QPushButton("一键运行")
        self._btn_clear = QPushButton("清空结果")

        for b in (self._btn_detect, self._btn_segment, self._btn_fusion, self._btn_full, self._btn_clear):
            b.setMinimumHeight(40)
            op_layout.addWidget(b)

        # 初始禁用：没有点云时禁止检测/分割/融合/一键
        self._btn_detect.setEnabled(False)
        self._btn_segment.setEnabled(False)
        self._btn_fusion.setEnabled(False)
        self._btn_full.setEnabled(False)
        self._btn_clear.setEnabled(True)

        root.addWidget(op_group)

        # 信号连接（面板自身转发为对外信号）
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

    # ----------------------------
    # 状态/数据读写（供控制器调用）
    # ----------------------------
    def set_selected_file(self, path: Optional[Path]) -> None:
        if path is None:
            self._label_file.setText("未选择")
            self._btn_load_file.setEnabled(False)
        else:
            self._label_file.setText(str(path))
            self._btn_load_file.setEnabled(True)

    def set_nusc_root(self, path: Optional[Path]) -> None:
        if path is None:
            self._label_nusc_root.setText("未选择")
            self._btn_nusc_connect.setEnabled(False)
        else:
            self._label_nusc_root.setText(str(path))
            self._btn_nusc_connect.setEnabled(True)

    def set_nusc_meta_text(self, text: str) -> None:
        self._label_nusc_meta.setText(str(text))

    def navigation_mode(self) -> str:
        return str(self._combo_mode.currentData())

    def set_scene_list(self, items: list[tuple[str, str]]) -> None:
        """
        items: [(scene_name, scene_token)]
        """
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
        self._combo_mode.setEnabled(enabled)
        mode = self.navigation_mode()
        self._combo_scene.setEnabled(enabled and mode == "scene")

        n = int(frame_count)
        self._spin_frame.setEnabled(enabled and n > 0)
        self._spin_frame.setMaximum(max(0, n - 1))
        if n <= 0:
            self._spin_frame.setValue(0)
        self._btn_prev.setEnabled(enabled and n > 0)
        self._btn_next.setEnabled(enabled and n > 0)
        self._btn_load_frame.setEnabled(enabled and n > 0)

    def frame_index(self) -> int:
        return int(self._spin_frame.value())

    def set_frame_index(self, idx: int) -> None:
        self._spin_frame.setValue(int(idx))

    def set_pointcloud_actions_enabled(self, has_pcd: bool, has_results: bool) -> None:
        self._btn_detect.setEnabled(bool(has_pcd))
        self._btn_segment.setEnabled(bool(has_pcd))
        # 融合显示依赖结果：至少有检测或分割结果之一
        self._btn_fusion.setEnabled(bool(has_pcd and has_results))
        self._btn_full.setEnabled(bool(has_pcd) or self._btn_load_file.isEnabled() or self._btn_nusc_connect.isEnabled())

    def set_busy(self, busy: bool) -> None:
        """
        busy=True：临时禁用可能触发重入的控件（清空结果仍可用）。
        busy=False：恢复到 busy 前的 enabled 状态。
        """
        lock_widgets = (
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

        if busy:
            self._enabled_cache = {id(w): bool(w.isEnabled()) for w in lock_widgets}
            for w in lock_widgets:
                w.setEnabled(False)
        else:
            for w in lock_widgets:
                prev = self._enabled_cache.get(id(w), True)
                w.setEnabled(bool(prev))
            self._enabled_cache = {}

