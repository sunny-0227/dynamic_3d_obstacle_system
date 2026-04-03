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
    QLineEdit,
    QPushButton,
    QSizePolicy,
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
        root.setSpacing(8)
        root.setContentsMargins(0, 0, 0, 0)

        # ---- 单文件 ----
        file_group = QGroupBox("本地点云文件")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(6)
        row_file = QHBoxLayout()
        self._btn_select_file = QPushButton("选择点云文件")
        self._path_file = QLineEdit()
        self._path_file.setReadOnly(True)
        self._path_file.setPlaceholderText("未选择")
        self._path_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._path_file.setMinimumHeight(28)
        row_file.addWidget(self._btn_select_file)
        row_file.addWidget(self._path_file, stretch=1)
        file_layout.addLayout(row_file)
        self._btn_load_file = QPushButton("加载点云")
        self._btn_load_file.setEnabled(False)
        self._btn_load_file.setMinimumHeight(36)
        file_layout.addWidget(self._btn_load_file)
        self._hint_single = QLabel(
            "单文件模式：选择 .bin/.pcd 后点击「加载点云」。连接 nuScenes 后此处将暂时禁用。"
        )
        self._hint_single.setWordWrap(True)
        self._hint_single.setStyleSheet("color: #6c7086; font-size: 12px;")
        file_layout.addWidget(self._hint_single)
        root.addWidget(file_group)

        # ---- nuScenes ----
        nusc_group = QGroupBox("nuScenes mini")
        nusc_layout = QVBoxLayout(nusc_group)
        nusc_layout.setSpacing(6)

        row_root = QHBoxLayout()
        self._btn_nusc_root = QPushButton("选择数据集根目录")
        self._path_nusc = QLineEdit()
        self._path_nusc.setReadOnly(True)
        self._path_nusc.setPlaceholderText("未选择")
        self._path_nusc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._path_nusc.setMinimumHeight(28)
        row_root.addWidget(self._btn_nusc_root)
        row_root.addWidget(self._path_nusc, stretch=1)
        self._btn_nusc_connect = QPushButton("加载数据集")
        self._btn_nusc_connect.setEnabled(False)
        self._btn_nusc_connect.setMinimumHeight(28)
        row_root.addWidget(self._btn_nusc_connect)
        nusc_layout.addLayout(row_root)

        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("导航方式:"))
        self._combo_mode = QComboBox()
        # 默认首项：与 NuScenesMiniLoader.connect() 内 set_navigation("global") 一致
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
        row_frame.addWidget(self._btn_prev)
        row_frame.addWidget(self._btn_next)
        row_frame.addWidget(self._btn_load_frame)
        nusc_layout.addLayout(row_frame)

        self._label_nusc_meta = QLabel("请先选择数据集根目录并点击「加载数据集」")
        self._label_nusc_meta.setWordWrap(True)
        self._label_nusc_meta.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        nusc_layout.addWidget(self._label_nusc_meta)

        self._hint_nusc = QLabel(
            "nuScenes：连接后默认使用「全数据集」导航并已生成帧列表，可直接选帧并点「加载当前帧点云」。"
            "若需按场景浏览，再切换「导航方式」即可。"
        )
        self._hint_nusc.setWordWrap(True)
        self._hint_nusc.setStyleSheet("color: #6c7086; font-size: 12px;")
        nusc_layout.addWidget(self._hint_nusc)

        root.addWidget(nusc_group)

        # ---- 操作按钮 ----
        op_group = QGroupBox("操作")
        op_layout = QVBoxLayout(op_group)
        op_layout.setSpacing(8)
        self._label_workflow = QLabel(
            "工作流状态：未选择（请使用上方「单文件」或「nuScenes」其一作为主流程）"
        )
        self._label_workflow.setWordWrap(True)
        self._label_workflow.setStyleSheet("color: #89b4fa; font-size: 12px;")
        op_layout.addWidget(self._label_workflow)
        row_ops = QHBoxLayout()
        row_ops.setSpacing(10)
        self._btn_detect = QPushButton("执行检测")
        self._btn_segment = QPushButton("执行分割")
        self._btn_fusion = QPushButton("融合显示")
        self._btn_full = QPushButton("一键运行")
        self._btn_clear = QPushButton("清空结果")

        for b in (self._btn_detect, self._btn_segment, self._btn_fusion, self._btn_full, self._btn_clear):
            b.setMinimumHeight(40)
            row_ops.addWidget(b)

        op_layout.addLayout(row_ops)

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

        self._busy_widgets = (
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

    # ----------------------------
    # 状态/数据读写（供控制器调用）
    # ----------------------------
    def set_selected_file(self, path: Optional[Path]) -> None:
        if path is None:
            self._path_file.clear()
            self._path_file.setPlaceholderText("未选择")
            self._btn_load_file.setEnabled(False)
        else:
            self._path_file.setText(str(path))
            # 若处于 nuScenes 锁定，由 apply_nusc_vs_single_file_lock 覆盖
            self._btn_load_file.setEnabled(True)

    def set_nusc_root(self, path: Optional[Path]) -> None:
        if path is None:
            self._path_nusc.clear()
            self._path_nusc.setPlaceholderText("未选择")
            self._btn_nusc_connect.setEnabled(False)
        else:
            self._path_nusc.setText(str(path))
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
        # 已选根目录但未连接时，也允许切换「导航方式」，便于在「加载数据集」前选好（默认仍为全数据集）
        root_set = bool(self._path_nusc.text().strip())
        self._combo_mode.setEnabled(enabled or root_set)
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

    def set_workflow_status_text(self, text: str) -> None:
        """操作区上方：工作流 / 数据集 / 帧 / 算法可用性摘要。"""
        self._label_workflow.setText(text)

    def apply_nusc_vs_single_file_lock(
        self,
        *,
        workflow: str,
        nusc_connected: bool,
    ) -> None:
        """
        nuScenes 已连接时禁用单文件入口；单文件已选路径时仍允许点「选择数据集根目录」切换（由主窗口弹窗确认）。
        """
        lock_single = workflow == "nuscenes" and nusc_connected
        self._btn_select_file.setEnabled(not lock_single)
        self._btn_load_file.setEnabled(not lock_single and self._path_file.text().strip() != "")
        self._hint_single.setVisible(not lock_single)

    def set_action_buttons_state(
        self,
        *,
        has_nonempty_pcd: bool,
        has_results: bool,
        allow_run_full_autoload: bool,
    ) -> None:
        """根据点云与单文件自动加载条件启用检测/分割/融合/一键。"""
        self._btn_detect.setEnabled(bool(has_nonempty_pcd))
        self._btn_segment.setEnabled(bool(has_nonempty_pcd))
        self._btn_fusion.setEnabled(bool(has_nonempty_pcd and has_results))
        self._btn_full.setEnabled(bool(has_nonempty_pcd or allow_run_full_autoload))

    def set_busy(self, busy: bool) -> None:
        """
        busy=True：临时禁用可能触发重入的控件（清空结果仍可用）。
        busy=False：恢复到 busy 前的 enabled 状态。
        """
        if busy:
            self._enabled_cache = {id(w): bool(w.isEnabled()) for w in self._busy_widgets}
            for w in self._busy_widgets:
                w.setEnabled(False)
        else:
            for w in self._busy_widgets:
                prev = self._enabled_cache.get(id(w), True)
                w.setEnabled(bool(prev))
            self._enabled_cache = {}

