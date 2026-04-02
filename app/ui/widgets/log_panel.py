"""
日志面板组件（里程碑 6）
用于答辩演示：显示运行日志，支持清空与追加。
"""

from __future__ import annotations

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout


class LogPanel(QGroupBox):
    def __init__(self, title: str = "运行日志", max_lines: int = 400, parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(int(max_lines))
        mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        if not mono_font or mono_font.family() == "":
            mono_font = QFont("Consolas", 9)
        else:
            mono_font.setPointSize(9)
        self._text.setFont(mono_font)
        layout.addWidget(self._text)

    def append(self, message: str) -> None:
        self._text.appendPlainText(str(message))

    def clear(self) -> None:
        self._text.clear()

