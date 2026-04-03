"""
运行日志面板（答辩版：加大可视区域、等宽字体、易读配色）
"""

from __future__ import annotations

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout


class LogPanel(QGroupBox):
    def __init__(self, title: str = "运行日志", max_lines: int = 800, parent=None):
        super().__init__(title, parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 12, 8, 8)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(int(max_lines))
        self._text.setMinimumHeight(320)
        mono_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        if not mono_font or mono_font.family() == "":
            mono_font = QFont("Consolas", 10)
        else:
            mono_font.setPointSize(10)
        self._text.setFont(mono_font)
        self._text.setPlaceholderText("操作与算法输出将显示在此处…")
        layout.addWidget(self._text)

    def append(self, message: str) -> None:
        self._text.appendPlainText(str(message))
        self._text.verticalScrollBar().setValue(self._text.verticalScrollBar().maximum())

    def clear(self) -> None:
        self._text.clear()
