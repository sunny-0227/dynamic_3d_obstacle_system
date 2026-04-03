"""
底部状态栏：数据集、当前帧/点云、任务状态（三分栏，科研系统常用信息布局）。
"""

from __future__ import annotations

from PyQt5.QtWidgets import QLabel, QStatusBar


class DefenseStatusBar(QStatusBar):
    """在一条状态栏内分区展示数据流与任务进度，便于实验复现与状态核对。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._lbl_dataset = QLabel("数据集：—")
        self._lbl_frame = QLabel("当前帧：—")
        self._lbl_exec = QLabel("任务状态：就绪")
        for w in (self._lbl_dataset, self._lbl_frame, self._lbl_exec):
            w.setMinimumWidth(160)
            w.setStyleSheet("color: #a6adc8; padding: 2px 6px;")
        self.addWidget(self._lbl_dataset, 2)
        self.addWidget(self._lbl_frame, 2)
        self.addWidget(self._lbl_exec, 3)

    def set_dataset_status(self, text: str) -> None:
        self._lbl_dataset.setText(str(text))

    def set_frame_status(self, text: str) -> None:
        self._lbl_frame.setText(str(text))

    def set_exec_status(self, text: str) -> None:
        self._lbl_exec.setText(str(text))
