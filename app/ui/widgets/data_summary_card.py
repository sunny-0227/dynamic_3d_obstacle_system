"""
答辩展示版右侧「当前数据摘要」卡片：结构化展示模式、路径、导航与帧信息。
"""

from __future__ import annotations

from PyQt5.QtGui import QFont, QFontDatabase
from PyQt5.QtWidgets import QGroupBox, QPlainTextEdit, QVBoxLayout


class DataSummaryCard(QGroupBox):
    """只读摘要区，使用等宽字体便于路径对齐（Windows 兼容）。"""

    def __init__(self, parent=None) -> None:
        super().__init__("当前数据摘要", parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 14, 10, 10)
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMinimumHeight(200)
        self._text.setMaximumHeight(280)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        if not mono or mono.family() == "":
            mono = QFont("Consolas", 10)
        else:
            mono.setPointSize(10)
        self._text.setFont(mono)
        layout.addWidget(self._text)

    def set_summary(self, text: str) -> None:
        self._text.setPlainText(str(text))
