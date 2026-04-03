"""
主窗口 QSS（科研系统深色主题，便于维护与复用）。
"""

from __future__ import annotations

# 集中管理样式字符串，避免主窗口类内大段字面量。
DEFENSE_MAINWINDOW_STYLESHEET = """
            QMainWindow { background-color: #11111b; }
            QWidget {
                background-color: #11111b;
                color: #cdd6f4;
                font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 13px;
            }
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
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #cba6f7;
                font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
            QPushButton {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton:pressed { background-color: #585b70; }
            QPushButton:disabled { color: #585b70; border-color: #313244; }
            QPushButton#btnOneClick {
                background-color: #45475a;
                color: #f5e0dc;
                border: 2px solid #89b4fa;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton#btnOneClick:hover { background-color: #585b70; }
            QPushButton#btnOneClick:disabled {
                color: #585b70;
                border-color: #313244;
                background-color: #313244;
            }
            QPlainTextEdit {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 6px;
                color: #cdd6f4;
                selection-background-color: #45475a;
            }
            QComboBox, QSpinBox, QLineEdit {
                background-color: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 4px 6px;
            }
            QScrollArea { background-color: transparent; border: none; }
            QStatusBar { background-color: #181825; border-top: 1px solid #313244; }
            QRadioButton { spacing: 6px; }
            QRadioButton::indicator { width: 14px; height: 14px; }
            """
