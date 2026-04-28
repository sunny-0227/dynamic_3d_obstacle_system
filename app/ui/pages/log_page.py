from __future__ import annotations

"""
系统日志页面

包含：
  - 运行日志显示区（可过滤）
  - 清空日志按钮
  - 导出日志按钮（导出到文件）
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LogPage(QWidget):
    """系统日志页面：接收日志追加请求，提供清空与导出功能。"""

    MAX_LINES = 2000   # 日志最多保留行数，超出自动截断

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._project_root: Optional[Path] = None
        self._build_ui()

    def set_project_root(self, root: Path) -> None:
        """设置项目根目录，导出时作为默认保存目录。"""
        self._project_root = root

    # ------------------------------------------------------------------
    # 构建 UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        # 【间距优化】外边距 28 左右、24 上下；行间距 16px
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        # 标题行 + 操作按钮
        title_row = QHBoxLayout()
        title = QLabel("系统日志")
        title.setObjectName("pageTitle")
        title_row.addWidget(title)
        title_row.addStretch(1)

        self._btn_clear  = QPushButton("清空日志")
        self._btn_export = QPushButton("导出日志")
        self._btn_clear.setMinimumHeight(36)   # 【间距优化】按钮高度 36px
        self._btn_export.setMinimumHeight(36)
        self._btn_clear.setMinimumWidth(90)
        self._btn_export.setMinimumWidth(90)
        title_row.addWidget(self._btn_clear)
        title_row.addWidget(self._btn_export)
        root.addLayout(title_row)

        root.addWidget(self._make_hline())

        # 行数统计标签
        self._lbl_count = QLabel("共 0 条记录")
        self._lbl_count.setObjectName("statusLabel")
        root.addWidget(self._lbl_count)

        # 日志文本区
        self._log_edit = QPlainTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setObjectName("logEdit")
        self._log_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        root.addWidget(self._log_edit, stretch=1)

        # 信号连接
        self._btn_clear.clicked.connect(self._on_clear)
        self._btn_export.clicked.connect(self._on_export)

    @staticmethod
    def _make_hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setObjectName("hline")
        return line

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def append(self, text: str) -> None:
        """追加一行日志（带时间戳）。"""
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_edit.appendPlainText(f"[{ts}] {text}")

        # 超出最大行数时截断头部
        doc = self._log_edit.document()
        while doc.blockCount() > self.MAX_LINES:
            cursor = self._log_edit.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()   # 删除换行符

        # 自动滚动到底部
        sb = self._log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())

        self._lbl_count.setText(f"共 {doc.blockCount()} 条记录")

    def clear(self) -> None:
        """清空日志内容。"""
        self._log_edit.clear()
        self._lbl_count.setText("共 0 条记录")

    def full_text(self) -> str:
        return self._log_edit.toPlainText()

    # ------------------------------------------------------------------
    # 内部槽
    # ------------------------------------------------------------------

    def _on_clear(self) -> None:
        self.clear()
        self.append("日志已清空")

    def _on_export(self) -> None:
        default_dir = str(self._project_root / "outputs" / "logs") if self._project_root else ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = Path(default_dir) / f"system_log_{ts}.txt"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出系统日志",
            str(default_name),
            "文本文件 (*.txt);;所有文件 (*)",
        )
        if not path:
            return
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(self.full_text(), encoding="utf-8")
            self.append(f"日志已导出到：{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"写入日志文件时出错：\n{e}")
