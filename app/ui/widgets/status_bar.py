"""
状态栏组件（里程碑 6）
封装 QStatusBar，便于控制器统一更新状态。
"""

from __future__ import annotations

from PyQt5.QtWidgets import QStatusBar


class StatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.showMessage("就绪")

    def set_status(self, message: str) -> None:
        self.showMessage(str(message))

