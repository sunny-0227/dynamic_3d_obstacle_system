"""
主窗口顶部标题区：课题主标题、副标题、工作模式与系统状态（科研/毕设风格文案）。
"""

from __future__ import annotations

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class DefenseHeader(QWidget):
    """顶栏信息条：突出正式标题，并展示当前工作模式与系统状态。"""

    def __init__(
        self,
        main_title: str,
        subtitle: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._lbl_main = QLabel(main_title)
        self._lbl_main.setObjectName("defenseAppTitle")
        self._lbl_sub = QLabel(subtitle)
        self._lbl_sub.setObjectName("defenseThesisTitle")
        self._lbl_sub.setWordWrap(True)
        self._lbl_mode = QLabel("工作模式：未选择数据源")
        self._lbl_mode.setObjectName("defenseModeLine")
        self._lbl_status = QLabel("系统状态：就绪")
        self._lbl_status.setObjectName("defenseStatusLine")

        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.addWidget(self._lbl_main)
        lay.addWidget(self._lbl_sub)
        lay.addWidget(self._lbl_mode)
        lay.addWidget(self._lbl_status)

    def set_mode_line(self, text: str) -> None:
        self._lbl_mode.setText(str(text))

    def set_status_line(self, text: str) -> None:
        self._lbl_status.setText(str(text))
