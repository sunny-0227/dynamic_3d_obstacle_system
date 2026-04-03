"""
答辩展示版顶部标题区：软件名、课题副标题、当前模式、当前状态。
"""

from __future__ import annotations

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class DefenseHeader(QWidget):
    """主窗口顶部信息条，便于答辩时一眼看到模式与状态。"""

    def __init__(
        self,
        app_name: str,
        thesis_title: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._lbl_app = QLabel(app_name)
        self._lbl_app.setObjectName("defenseAppTitle")
        self._lbl_thesis = QLabel(thesis_title)
        self._lbl_thesis.setObjectName("defenseThesisTitle")
        self._lbl_thesis.setWordWrap(True)
        self._lbl_mode = QLabel("当前模式：未选择数据源")
        self._lbl_mode.setObjectName("defenseModeLine")
        self._lbl_status = QLabel("当前状态：就绪")
        self._lbl_status.setObjectName("defenseStatusLine")

        lay = QVBoxLayout(self)
        lay.setSpacing(4)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.addWidget(self._lbl_app)
        lay.addWidget(self._lbl_thesis)
        lay.addWidget(self._lbl_mode)
        lay.addWidget(self._lbl_status)

    def set_mode_line(self, text: str) -> None:
        self._lbl_mode.setText(str(text))

    def set_status_line(self, text: str) -> None:
        self._lbl_status.setText(str(text))
