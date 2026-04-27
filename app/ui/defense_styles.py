"""
主窗口 QSS（科研系统深色主题，便于维护与复用）。

颜色体系（Catppuccin Mocha 派生）：
  背景层      #11111b / #181825 / #1e1e2e
  边框        #313244 / #45475a / #585b70
  主文字      #cdd6f4
  次要文字    #a6adc8 / #6c7086
  强调蓝      #89b4fa
  强调紫      #cba6f7
  成功绿      #a6e3a1
  警告黄      #f9e2af
  错误红      #f38ba8
  温暖白      #f5e0dc
"""

from __future__ import annotations

DEFENSE_MAINWINDOW_STYLESHEET = """
/* ──────────────── 基础 ──────────────── */
QMainWindow {
    background-color: #11111b;
}
QWidget {
    background-color: #11111b;
    color: #cdd6f4;
    font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
}

/* ──────────────── 导航栏 ──────────────── */
QWidget#navBar {
    background-color: #181825;
    border-right: 1px solid #313244;
}
QLabel#sysTitle {
    font-size: 13px;
    font-weight: bold;
    color: #89b4fa;
    padding: 0 12px;
}
QLabel#sysSubtitle {
    font-size: 10px;
    color: #6c7086;
    padding: 0 12px;
}
QPushButton#navBtn {
    background-color: transparent;
    color: #a6adc8;
    border: none;
    border-radius: 0;
    text-align: left;
    padding: 12px 18px;
    font-size: 13px;
}
QPushButton#navBtn:hover {
    background-color: #1e1e2e;
    color: #cdd6f4;
}
QPushButton#navBtn[active="true"] {
    background-color: #313244;
    color: #89b4fa;
    border-left: 3px solid #89b4fa;
    font-weight: bold;
}

/* ──────────────── 顶栏 ──────────────── */
QLabel#mainTitle {
    font-size: 18px;
    font-weight: bold;
    color: #89b4fa;
    padding: 2px 0;
}
QLabel#subTitle {
    font-size: 12px;
    color: #a6adc8;
}
QLabel#statusLine {
    font-size: 12px;
    color: #a6e3a1;
}

/* ──────────────── 兼容旧 defenseHeader 控件名 ──────────────── */
QLabel#defenseAppTitle {
    font-size: 20px;
    font-weight: bold;
    color: #89b4fa;
}
QLabel#defenseThesisTitle {
    font-size: 12px;
    color: #a6adc8;
}
QLabel#defenseModeLine {
    font-size: 13px;
    color: #f5e0dc;
    font-weight: bold;
}
QLabel#defenseStatusLine {
    font-size: 12px;
    color: #a6e3a1;
}

/* ──────────────── 页面标题 ──────────────── */
QLabel#pageTitle {
    font-size: 16px;
    font-weight: bold;
    color: #cba6f7;
    padding: 4px 0 8px 0;
}

/* ──────────────── 状态标签 ──────────────── */
QLabel#statusLabel {
    color: #a6adc8;
    font-size: 12px;
}
QLabel#statusKey {
    color: #6c7086;
    font-size: 12px;
}
QLabel#statusValue {
    color: #cdd6f4;
    font-size: 12px;
    font-weight: bold;
}
QLabel#hintLabel {
    color: #6c7086;
    font-size: 11px;
}
QLabel#metaLabel {
    color: #a6adc8;
    font-size: 11px;
}

/* ──────────────── 分组框 ──────────────── */
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    color: #cba6f7;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

/* ──────────────── 按钮 ──────────────── */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #585b70;
    border-radius: 6px;
    padding: 6px 12px;
}
QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}
QPushButton:pressed {
    background-color: #585b70;
}
QPushButton:disabled {
    color: #585b70;
    border-color: #313244;
    background-color: #1e1e2e;
}
/* 一键分析（突出主操作按钮） */
QPushButton#btnOneClick {
    background-color: #45475a;
    color: #f5e0dc;
    border: 2px solid #89b4fa;
    font-weight: bold;
    font-size: 14px;
}
QPushButton#btnOneClick:hover {
    background-color: #585b70;
}
QPushButton#btnOneClick:disabled {
    color: #585b70;
    border-color: #313244;
    background-color: #313244;
}

/* ──────────────── 文本输入 ──────────────── */
QPlainTextEdit {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 6px;
    color: #cdd6f4;
    selection-background-color: #45475a;
}
QPlainTextEdit#logEdit {
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 4px 6px;
}
QComboBox::drop-down {
    border: none;
}

/* ──────────────── 滚动条 ──────────────── */
QScrollBar:vertical {
    background: #181825;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #585b70;
}
QScrollBar:horizontal {
    background: #181825;
    height: 8px;
    border-radius: 4px;
}
QScrollBar::handle:horizontal {
    background: #45475a;
    border-radius: 4px;
    min-width: 20px;
}
QScrollArea {
    background-color: transparent;
    border: none;
}

/* ──────────────── 单选/复选 ──────────────── */
QRadioButton, QCheckBox {
    spacing: 8px;
    color: #cdd6f4;
}
QRadioButton::indicator, QCheckBox::indicator {
    width: 14px;
    height: 14px;
}

/* ──────────────── 分割线 ──────────────── */
QFrame#hline {
    color: #313244;
}

/* ──────────────── 状态栏 ──────────────── */
QStatusBar {
    background-color: #181825;
    border-top: 1px solid #313244;
    color: #a6adc8;
    font-size: 11px;
}
"""
