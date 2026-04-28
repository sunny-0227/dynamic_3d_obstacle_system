from __future__ import annotations

"""
模型与配置页面

包含：
  - OpenPCDet WSL 配置（infer_script / cfg_file / ckpt_file / conda_env）
  - RealSense 参数（分辨率、帧率、深度范围）
  - 轻量分割参数（RANSAC 阈值、迭代次数）
  - 聚类参数（DBSCAN eps / min_samples）
  - 保存配置按钮（写回 settings.yaml）

布局说明：
  - 整体使用 QScrollArea 包裹，防止窗口过小时内容被截断
  - 左右两列均等分配（stretch=1:1）
  - 每个 GroupBox 内部行间距 12px，内边距 16/22/16/18px
  - 路径行：标签在输入框上方，提高横向可读性
  - 数值行：标签固定宽度 190px，控件向右拉伸
  - GroupBox 之间间距 20px
  - 整体外边距 28/20（左右/上下）
"""

from pathlib import Path
from typing import Optional

import yaml
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# 数值行标签固定宽度（保证所有 SpinBox 纵向对齐）
_LABEL_WIDTH = 190


class ConfigPage(QWidget):
    """模型与配置页面，用于展示与修改运行参数，支持保存至 settings.yaml。"""

    sig_config_saved = pyqtSignal()

    def __init__(self, config: dict, project_root: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self._project_root = project_root
        self._settings_file = project_root / "config" / "settings.yaml"
        self._build_ui()
        self._load_from_config()

    # ------------------------------------------------------------------
    # 构建 UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # 最外层：页面级 VBox，包含标题 + 滚动区 + 保存按钮
        page_lay = QVBoxLayout(self)
        page_lay.setContentsMargins(0, 0, 0, 0)
        page_lay.setSpacing(0)

        # ── 标题行（不随滚动移动）────────────────────────────────────
        title_wrap = QWidget()
        title_wrap.setObjectName("titleWrap")
        tw_lay = QVBoxLayout(title_wrap)
        tw_lay.setContentsMargins(28, 20, 28, 12)
        tw_lay.setSpacing(8)
        title = QLabel("模型与配置")
        title.setObjectName("pageTitle")
        tw_lay.addWidget(title)
        tw_lay.addWidget(self._make_hline())
        page_lay.addWidget(title_wrap)

        # ── 可滚动内容区 ─────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)          # 去掉滚动区边框
        scroll.setHorizontalScrollBarPolicy(scroll.horizontalScrollBarPolicy())

        content_widget = QWidget()
        content_lay = QVBoxLayout(content_widget)
        content_lay.setContentsMargins(28, 16, 28, 16)
        content_lay.setSpacing(16)

        # 左右分栏
        body = QHBoxLayout()
        body.setSpacing(28)                           # 两列之间留 28px
        body.addLayout(self._build_left_col(),  stretch=1)
        body.addLayout(self._build_right_col(), stretch=1)
        content_lay.addLayout(body)
        content_lay.addStretch(1)

        scroll.setWidget(content_widget)
        page_lay.addWidget(scroll, stretch=1)

        # ── 保存区（固定在底部）──────────────────────────────────────
        bottom_wrap = QWidget()
        bot_lay = QVBoxLayout(bottom_wrap)
        bot_lay.setContentsMargins(28, 10, 28, 14)
        bot_lay.setSpacing(6)
        bot_lay.addWidget(self._make_hline())

        save_row = QHBoxLayout()
        self._btn_save = QPushButton("保存配置到 config/settings.yaml")
        self._btn_save.setObjectName("btnOneClick")
        self._btn_save.setMinimumHeight(48)
        self._btn_save.setMinimumWidth(340)
        self._btn_save.clicked.connect(self._on_save)
        save_row.addStretch(1)
        save_row.addWidget(self._btn_save)
        save_row.addStretch(1)
        bot_lay.addLayout(save_row)

        self._lbl_save_hint = QLabel("")
        self._lbl_save_hint.setObjectName("hintLabel")
        self._lbl_save_hint.setMinimumHeight(20)
        bot_lay.addWidget(self._lbl_save_hint)
        page_lay.addWidget(bottom_wrap)

    # ── 左列：OpenPCDet + RealSense ──────────────────────────────────

    def _build_left_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(20)                            # GroupBox 间距 20px

        # ── OpenPCDet WSL ─────────────────────────────────────────────
        opc_group = QGroupBox("OpenPCDet（WSL2 推理）")
        opc_lay = QVBoxLayout(opc_group)
        opc_lay.setSpacing(12)                        # 行间距 12px（从 10 放宽）
        opc_lay.setContentsMargins(16, 22, 16, 18)    # 内边距四向加大

        self._chk_enable_wsl = QCheckBox("启用 OpenPCDet 真实检测（通过 WSL2）")
        self._chk_enable_wsl.setMinimumHeight(30)
        opc_lay.addWidget(self._chk_enable_wsl)

        opc_lay.addSpacing(6)
        opc_lay.addWidget(self._make_hline())
        opc_lay.addSpacing(6)

        self._le_infer_script     = self._make_path_row(opc_lay, "推理脚本路径（WSL）：")
        self._le_cfg_file         = self._make_path_row(opc_lay, "模型配置文件（WSL）：")
        self._le_ckpt_file        = self._make_path_row(opc_lay, "模型权重文件（WSL）：")
        self._le_tools_dir        = self._make_path_row(opc_lay, "OpenPCDet tools 目录（WSL）：")
        self._le_conda_env        = self._make_path_row(opc_lay, "Conda 环境名称：")
        self._le_tmp_dir          = self._make_path_row(opc_lay, "临时输出目录（Win）：")

        opc_lay.addSpacing(6)
        hint = QLabel("⚠ 修改路径后需重启应用，检测器才会重新初始化。")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        hint.setMinimumHeight(22)
        opc_lay.addWidget(hint)
        col.addWidget(opc_group)

        # ── RealSense 参数 ─────────────────────────────────────────────
        rs_group = QGroupBox("Intel RealSense 参数")
        rs_lay = QVBoxLayout(rs_group)
        rs_lay.setSpacing(12)
        rs_lay.setContentsMargins(16, 22, 16, 18)

        self._spin_rs_width   = self._make_spin_row(
            rs_lay, "深度图宽度（像素）：",   128, 1920, 8)
        self._spin_rs_height  = self._make_spin_row(
            rs_lay, "深度图高度（像素）：",   96,  1080, 8)
        self._spin_rs_fps     = self._make_spin_row(
            rs_lay, "深度帧率（FPS）：",      6,   90,   1)
        self._dspin_min_depth = self._make_dspin_row(
            rs_lay, "最小有效深度（米）：",   0.01, 5.0,  0.01)
        self._dspin_max_depth = self._make_dspin_row(
            rs_lay, "最大有效深度（米）：",   1.0,  30.0, 0.5)

        rs_lay.addSpacing(4)
        hint_rs = QLabel("建议分辨率：424×240（约 5 万点，处理速度快）")
        hint_rs.setObjectName("hintLabel")
        hint_rs.setWordWrap(True)
        hint_rs.setMinimumHeight(22)
        rs_lay.addWidget(hint_rs)
        col.addWidget(rs_group)

        col.addStretch(1)
        return col

    # ── 右列：RANSAC + DBSCAN ─────────────────────────────────────────

    def _build_right_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(20)

        # ── 轻量分割参数 ──────────────────────────────────────────────
        seg_group = QGroupBox("轻量分割参数（RANSAC 地面检测）")
        seg_lay = QVBoxLayout(seg_group)
        seg_lay.setSpacing(12)
        seg_lay.setContentsMargins(16, 22, 16, 18)

        self._dspin_ransac_thresh  = self._make_dspin_row(
            seg_lay, "RANSAC 距离阈值（米）：",  0.01, 0.5,  0.01)
        self._spin_ransac_iters    = self._make_spin_row(
            seg_lay, "RANSAC 最大迭代次数：",    10,   2000, 1)
        self._dspin_obstacle_min_h = self._make_dspin_row(
            seg_lay, "障碍物最低高度（米）：",   0.05, 2.0,  0.05)
        self._dspin_obstacle_max_h = self._make_dspin_row(
            seg_lay, "障碍物最高高度（米）：",   0.5,  10.0, 0.1)

        seg_lay.addSpacing(4)
        hint_seg = QLabel("距离阈值控制地面分割精度；迭代次数影响速度与准确性。")
        hint_seg.setObjectName("hintLabel")
        hint_seg.setWordWrap(True)
        hint_seg.setMinimumHeight(22)
        seg_lay.addWidget(hint_seg)
        col.addWidget(seg_group)

        # ── 聚类参数 ──────────────────────────────────────────────────
        cls_group = QGroupBox("聚类参数（DBSCAN）")
        cls_lay = QVBoxLayout(cls_group)
        cls_lay.setSpacing(12)
        cls_lay.setContentsMargins(16, 22, 16, 18)

        self._dspin_dbscan_eps     = self._make_dspin_row(
            cls_lay, "DBSCAN eps（邻域半径，米）：", 0.05, 5.0,   0.05)
        self._spin_dbscan_min_pts  = self._make_spin_row(
            cls_lay, "DBSCAN min_samples：",         1,    100,   1)
        self._spin_cluster_min_pts = self._make_spin_row(
            cls_lay, "聚类保留最小点数：",            3,    500,   1)
        self._spin_cluster_max_pts = self._make_spin_row(
            cls_lay, "聚类保留最大点数：",            10,   50000, 10)

        cls_lay.addSpacing(4)
        hint_cls = QLabel("点数少于【最小点数】或多于【最大点数】的聚类将被过滤。")
        hint_cls.setObjectName("hintLabel")
        hint_cls.setWordWrap(True)
        hint_cls.setMinimumHeight(22)
        cls_lay.addWidget(hint_cls)
        col.addWidget(cls_group)

        col.addStretch(1)
        return col

    # ------------------------------------------------------------------
    # 辅助控件构建
    # ------------------------------------------------------------------

    @staticmethod
    def _make_path_row(parent_lay: QVBoxLayout, label: str) -> QLineEdit:
        """路径输入行：标签在上，输入框在下，避免横向空间不足时被压缩。"""
        lbl = QLabel(label)
        lbl.setMinimumHeight(22)
        parent_lay.addWidget(lbl)
        le = QLineEdit()
        le.setPlaceholderText("（未配置）")
        le.setMinimumHeight(34)          # 高度比之前 32px 略大，更易点击
        le.setMinimumWidth(220)
        parent_lay.addWidget(le)
        return le

    @staticmethod
    def _make_spin_row(
        parent_lay: QVBoxLayout, label: str, min_v: int, max_v: int, step: int
    ) -> QSpinBox:
        """整数输入行：标签固定宽度，SpinBox 向右拉伸。"""
        row = QHBoxLayout()
        row.setSpacing(12)
        lbl = QLabel(label)
        lbl.setFixedWidth(_LABEL_WIDTH)
        lbl.setMinimumHeight(26)
        spin = QSpinBox()
        spin.setMinimum(min_v)
        spin.setMaximum(max_v)
        spin.setSingleStep(step)
        spin.setMinimumHeight(34)
        spin.setMinimumWidth(120)
        row.addWidget(lbl)
        row.addWidget(spin, stretch=1)
        parent_lay.addLayout(row)
        return spin

    @staticmethod
    def _make_dspin_row(
        parent_lay: QVBoxLayout, label: str, min_v: float, max_v: float, step: float
    ) -> QDoubleSpinBox:
        """浮点输入行：标签固定宽度，DoubleSpin 向右拉伸。"""
        row = QHBoxLayout()
        row.setSpacing(12)
        lbl = QLabel(label)
        lbl.setFixedWidth(_LABEL_WIDTH)
        lbl.setMinimumHeight(26)
        spin = QDoubleSpinBox()
        spin.setMinimum(min_v)
        spin.setMaximum(max_v)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setMinimumHeight(34)
        spin.setMinimumWidth(120)
        row.addWidget(lbl)
        row.addWidget(spin, stretch=1)
        parent_lay.addLayout(row)
        return spin

    @staticmethod
    def _make_hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setObjectName("hline")
        return line

    # ------------------------------------------------------------------
    # 从配置字典加载
    # ------------------------------------------------------------------

    def _load_from_config(self) -> None:
        wsl_cfg = self._config.get("detector", {}).get("openpcdet_wsl", {})
        self._chk_enable_wsl.setChecked(bool(wsl_cfg.get("enable_wsl", False)))
        self._le_infer_script.setText(str(wsl_cfg.get("infer_script", "")))
        self._le_cfg_file.setText(str(wsl_cfg.get("cfg_file", "")))
        self._le_ckpt_file.setText(str(wsl_cfg.get("ckpt_file", "")))
        self._le_tools_dir.setText(str(wsl_cfg.get("openpcdet_tools_dir", "")))
        self._le_conda_env.setText(str(wsl_cfg.get("conda_env", "")))
        self._le_tmp_dir.setText(str(wsl_cfg.get("tmp_dir", "")))

        rt_cfg = self._config.get("realtime", {})
        # 兼容新键名（camera_width/camera_height/camera_fps/min_depth/max_depth）
        # 和旧键名（width/height/fps/min_depth_m/max_depth_m），新键名优先
        self._spin_rs_width.setValue(
            int(rt_cfg.get("camera_width",  rt_cfg.get("width",       424))))
        self._spin_rs_height.setValue(
            int(rt_cfg.get("camera_height", rt_cfg.get("height",      240))))
        self._spin_rs_fps.setValue(
            int(rt_cfg.get("camera_fps",    rt_cfg.get("fps",         30))))
        self._dspin_min_depth.setValue(
            float(rt_cfg.get("min_depth",   rt_cfg.get("min_depth_m", 0.3))))
        self._dspin_max_depth.setValue(
            float(rt_cfg.get("max_depth",   rt_cfg.get("max_depth_m", 4.0))))

        seg_cfg = self._config.get("realtime_segmentor", {})
        self._dspin_ransac_thresh.setValue(
            float(seg_cfg.get("ransac_dist_threshold", 0.2)))
        self._spin_ransac_iters.setValue(
            int(seg_cfg.get("ransac_max_iters", 300)))
        self._dspin_obstacle_min_h.setValue(
            float(seg_cfg.get("obstacle_min_height", 0.2)))
        self._dspin_obstacle_max_h.setValue(
            float(seg_cfg.get("obstacle_max_height", 3.0)))

        det_cfg = self._config.get("realtime_detector", {})
        self._dspin_dbscan_eps.setValue(
            float(det_cfg.get("dbscan_eps", 0.5)))
        self._spin_dbscan_min_pts.setValue(
            int(det_cfg.get("dbscan_min_samples", 5)))
        self._spin_cluster_min_pts.setValue(
            int(det_cfg.get("cluster_min_pts", 10)))
        self._spin_cluster_max_pts.setValue(
            int(det_cfg.get("cluster_max_pts", 5000)))

    # ------------------------------------------------------------------
    # 保存到文件
    # ------------------------------------------------------------------

    def _on_save(self) -> None:
        """将表单值写入 settings.yaml。"""
        try:
            with open(self._settings_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # --- OpenPCDet WSL ---
            if "detector" not in data:
                data["detector"] = {}
            if "openpcdet_wsl" not in data["detector"]:
                data["detector"]["openpcdet_wsl"] = {}
            wsl = data["detector"]["openpcdet_wsl"]
            wsl["enable_wsl"]            = self._chk_enable_wsl.isChecked()
            wsl["infer_script"]          = self._le_infer_script.text().strip()
            wsl["cfg_file"]              = self._le_cfg_file.text().strip()
            wsl["ckpt_file"]             = self._le_ckpt_file.text().strip()
            wsl["openpcdet_tools_dir"]   = self._le_tools_dir.text().strip()
            wsl["conda_env"]             = self._le_conda_env.text().strip()
            wsl["tmp_dir"]               = self._le_tmp_dir.text().strip()

            # --- RealSense（新旧键名同时写，确保各处均可读到）---
            if "realtime" not in data:
                data["realtime"] = {}
            rt = data["realtime"]
            rt["width"]         = self._spin_rs_width.value()
            rt["height"]        = self._spin_rs_height.value()
            rt["fps"]           = self._spin_rs_fps.value()
            rt["camera_width"]  = self._spin_rs_width.value()
            rt["camera_height"] = self._spin_rs_height.value()
            rt["camera_fps"]    = self._spin_rs_fps.value()
            rt["min_depth_m"]   = round(self._dspin_min_depth.value(), 3)
            rt["max_depth_m"]   = round(self._dspin_max_depth.value(), 3)
            rt["min_depth"]     = round(self._dspin_min_depth.value(), 3)
            rt["max_depth"]     = round(self._dspin_max_depth.value(), 3)

            # --- 轻量分割参数 ---
            if "realtime_segmentor" not in data:
                data["realtime_segmentor"] = {}
            seg = data["realtime_segmentor"]
            seg["ransac_dist_threshold"] = round(self._dspin_ransac_thresh.value(), 3)
            seg["ransac_max_iters"]      = self._spin_ransac_iters.value()
            seg["obstacle_min_height"]   = round(self._dspin_obstacle_min_h.value(), 3)
            seg["obstacle_max_height"]   = round(self._dspin_obstacle_max_h.value(), 3)

            # --- DBSCAN 聚类参数 ---
            if "realtime_detector" not in data:
                data["realtime_detector"] = {}
            det = data["realtime_detector"]
            det["dbscan_eps"]         = round(self._dspin_dbscan_eps.value(), 3)
            det["dbscan_min_samples"] = self._spin_dbscan_min_pts.value()
            det["cluster_min_pts"]    = self._spin_cluster_min_pts.value()
            det["cluster_max_pts"]    = self._spin_cluster_max_pts.value()

            with open(self._settings_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True,
                          default_flow_style=False, sort_keys=False)

            self._lbl_save_hint.setText(
                "✓ 配置已保存。请重启应用以使部分参数（如检测器路径）生效。"
            )
            self.sig_config_saved.emit()

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入 settings.yaml 时出错：\n{e}")
