from __future__ import annotations

"""
模型与配置页面

包含：
  - OpenPCDet WSL 配置（infer_script / cfg_file / ckpt_file / conda_env）
  - 是否启用 OpenPCDet
  - RealSense 参数（分辨率、帧率、深度范围）
  - 轻量分割参数（RANSAC 阈值、迭代次数）
  - 聚类参数（DBSCAN eps / min_samples）
  - 保存配置按钮（写回 settings.yaml）
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ConfigPage(QWidget):
    """模型与配置页面，用于展示与修改运行参数，支持保存至 settings.yaml。"""

    sig_config_saved = pyqtSignal()   # 保存完成后通知主窗口重新加载配置

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
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        title = QLabel("模型与配置")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(self._make_hline())

        # 两列布局
        body = QHBoxLayout()
        body.setSpacing(16)
        body.addLayout(self._build_left_col(), stretch=1)
        body.addLayout(self._build_right_col(), stretch=1)
        root.addLayout(body)

        root.addWidget(self._make_hline())

        # 保存按钮
        save_row = QHBoxLayout()
        self._btn_save = QPushButton("保存配置到 config/settings.yaml")
        self._btn_save.setObjectName("btnOneClick")
        self._btn_save.setMinimumHeight(44)
        self._btn_save.clicked.connect(self._on_save)
        save_row.addStretch(1)
        save_row.addWidget(self._btn_save)
        save_row.addStretch(1)
        root.addLayout(save_row)

        self._lbl_save_hint = QLabel("")
        self._lbl_save_hint.setObjectName("hintLabel")
        root.addWidget(self._lbl_save_hint)

        root.addStretch(1)

    def _build_left_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(12)

        # ── OpenPCDet WSL ─────────────────────────────────────────
        opc_group = QGroupBox("OpenPCDet（WSL2 推理）")
        opc_lay = QVBoxLayout(opc_group)
        opc_lay.setSpacing(8)

        self._chk_enable_wsl = QCheckBox("启用 OpenPCDet 真实检测（通过 WSL2）")
        opc_lay.addWidget(self._chk_enable_wsl)

        opc_lay.addWidget(self._make_hline())

        self._le_infer_script = self._make_label_edit(opc_lay, "推理脚本路径（WSL 内）：")
        self._le_cfg_file     = self._make_label_edit(opc_lay, "模型配置文件（WSL 内）：")
        self._le_ckpt_file    = self._make_label_edit(opc_lay, "模型权重文件（WSL 内）：")
        self._le_conda_env    = self._make_label_edit(opc_lay, "Conda 环境名称：")
        self._le_tmp_dir      = self._make_label_edit(opc_lay, "临时输出目录（Windows）：")

        hint = QLabel("⚠ 修改后需重启应用以使检测器重新初始化。")
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        opc_lay.addWidget(hint)
        col.addWidget(opc_group)

        # ── RealSense 参数 ────────────────────────────────────────
        rs_group = QGroupBox("Intel RealSense 参数")
        rs_lay = QVBoxLayout(rs_group)
        rs_lay.setSpacing(8)

        self._spin_rs_width    = self._make_spin(rs_lay, "深度图宽度（像素）：", 128, 1920, 8)
        self._spin_rs_height   = self._make_spin(rs_lay, "深度图高度（像素）：", 96, 1080, 8)
        self._spin_rs_fps      = self._make_spin(rs_lay, "深度帧率（FPS）：",    6,   90,  1)
        self._dspin_min_depth  = self._make_dspin(rs_lay, "最小有效深度（米）：", 0.01, 5.0,  0.01)
        self._dspin_max_depth  = self._make_dspin(rs_lay, "最大有效深度（米）：", 1.0,  30.0, 0.5)
        col.addWidget(rs_group)

        col.addStretch(1)
        return col

    def _build_right_col(self) -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(12)

        # ── 轻量分割参数 ─────────────────────────────────────────
        seg_group = QGroupBox("轻量分割参数（RANSAC 地面检测）")
        seg_lay = QVBoxLayout(seg_group)
        seg_lay.setSpacing(8)

        self._dspin_ransac_thresh  = self._make_dspin(seg_lay, "RANSAC 距离阈值（米）：",  0.01, 0.5,  0.01)
        self._spin_ransac_iters    = self._make_spin(seg_lay,  "RANSAC 最大迭代次数：",    10,   2000, 1)
        self._dspin_obstacle_min_h = self._make_dspin(seg_lay, "障碍物最低高度（地面以上，米）：", 0.05, 2.0, 0.05)
        self._dspin_obstacle_max_h = self._make_dspin(seg_lay, "障碍物最高高度（地面以上，米）：", 0.5,  10.0, 0.1)
        col.addWidget(seg_group)

        # ── 聚类参数 ─────────────────────────────────────────────
        cls_group = QGroupBox("聚类参数（DBSCAN）")
        cls_lay = QVBoxLayout(cls_group)
        cls_lay.setSpacing(8)

        self._dspin_dbscan_eps     = self._make_dspin(cls_lay, "DBSCAN eps（邻域半径，米）：",  0.05, 5.0,  0.05)
        self._spin_dbscan_min_pts  = self._make_spin(cls_lay,  "DBSCAN min_samples：",          1,    100,  1)
        self._spin_cluster_min_pts = self._make_spin(cls_lay,  "聚类保留最小点数：",             3,    500,  1)
        self._spin_cluster_max_pts = self._make_spin(cls_lay,  "聚类保留最大点数：",             10,   50000, 10)

        hint_cls = QLabel("点数少于【最小点数】或多于【最大点数】的聚类将被过滤。")
        hint_cls.setObjectName("hintLabel")
        hint_cls.setWordWrap(True)
        cls_lay.addWidget(hint_cls)
        col.addWidget(cls_group)

        col.addStretch(1)
        return col

    # ------------------------------------------------------------------
    # 辅助控件构建
    # ------------------------------------------------------------------

    @staticmethod
    def _make_label_edit(parent_lay: QVBoxLayout, label: str) -> QLineEdit:
        parent_lay.addWidget(QLabel(label))
        le = QLineEdit()
        le.setPlaceholderText("（未配置）")
        parent_lay.addWidget(le)
        return le

    @staticmethod
    def _make_spin(
        parent_lay: QVBoxLayout, label: str, min_v: int, max_v: int, step: int
    ) -> QSpinBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(240)
        spin = QSpinBox()
        spin.setMinimum(min_v)
        spin.setMaximum(max_v)
        spin.setSingleStep(step)
        row.addWidget(lbl)
        row.addWidget(spin, stretch=1)
        parent_lay.addLayout(row)
        return spin

    @staticmethod
    def _make_dspin(
        parent_lay: QVBoxLayout, label: str, min_v: float, max_v: float, step: float
    ) -> QDoubleSpinBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setFixedWidth(240)
        spin = QDoubleSpinBox()
        spin.setMinimum(min_v)
        spin.setMaximum(max_v)
        spin.setSingleStep(step)
        spin.setDecimals(3)
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
        self._le_conda_env.setText(str(wsl_cfg.get("conda_env", "")))
        self._le_tmp_dir.setText(str(wsl_cfg.get("tmp_dir", "")))

        rt_cfg = self._config.get("realtime", {})
        self._spin_rs_width.setValue(int(rt_cfg.get("width", 640)))
        self._spin_rs_height.setValue(int(rt_cfg.get("height", 480)))
        self._spin_rs_fps.setValue(int(rt_cfg.get("fps", 30)))
        self._dspin_min_depth.setValue(float(rt_cfg.get("min_depth_m", 0.1)))
        self._dspin_max_depth.setValue(float(rt_cfg.get("max_depth_m", 10.0)))

        seg_cfg = self._config.get("realtime_segmentor", {})
        self._dspin_ransac_thresh.setValue(float(seg_cfg.get("ransac_dist_threshold", 0.2)))
        self._spin_ransac_iters.setValue(int(seg_cfg.get("ransac_max_iters", 300)))
        self._dspin_obstacle_min_h.setValue(float(seg_cfg.get("obstacle_min_height", 0.2)))
        self._dspin_obstacle_max_h.setValue(float(seg_cfg.get("obstacle_max_height", 3.0)))

        det_cfg = self._config.get("realtime_detector", {})
        self._dspin_dbscan_eps.setValue(float(det_cfg.get("dbscan_eps", 0.5)))
        self._spin_dbscan_min_pts.setValue(int(det_cfg.get("dbscan_min_samples", 5)))
        self._spin_cluster_min_pts.setValue(int(det_cfg.get("cluster_min_pts", 10)))
        self._spin_cluster_max_pts.setValue(int(det_cfg.get("cluster_max_pts", 5000)))

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
            wsl["enable_wsl"]    = self._chk_enable_wsl.isChecked()
            wsl["infer_script"]  = self._le_infer_script.text().strip()
            wsl["cfg_file"]      = self._le_cfg_file.text().strip()
            wsl["ckpt_file"]     = self._le_ckpt_file.text().strip()
            wsl["conda_env"]     = self._le_conda_env.text().strip()
            wsl["tmp_dir"]       = self._le_tmp_dir.text().strip()

            # --- RealSense ---
            if "realtime" not in data:
                data["realtime"] = {}
            rt = data["realtime"]
            rt["width"]       = self._spin_rs_width.value()
            rt["height"]      = self._spin_rs_height.value()
            rt["fps"]         = self._spin_rs_fps.value()
            rt["min_depth_m"] = round(self._dspin_min_depth.value(), 3)
            rt["max_depth_m"] = round(self._dspin_max_depth.value(), 3)

            # --- 轻量分割参数（写入 realtime_segmentor 子块） ---
            if "realtime_segmentor" not in data:
                data["realtime_segmentor"] = {}
            seg = data["realtime_segmentor"]
            seg["ransac_dist_threshold"] = round(self._dspin_ransac_thresh.value(), 3)
            seg["ransac_max_iters"]      = self._spin_ransac_iters.value()
            seg["obstacle_min_height"]   = round(self._dspin_obstacle_min_h.value(), 3)
            seg["obstacle_max_height"]   = round(self._dspin_obstacle_max_h.value(), 3)

            # --- DBSCAN 聚类参数（写入 realtime_detector 子块） ---
            if "realtime_detector" not in data:
                data["realtime_detector"] = {}
            det = data["realtime_detector"]
            det["dbscan_eps"]        = round(self._dspin_dbscan_eps.value(), 3)
            det["dbscan_min_samples"] = self._spin_dbscan_min_pts.value()
            det["cluster_min_pts"]   = self._spin_cluster_min_pts.value()
            det["cluster_max_pts"]   = self._spin_cluster_max_pts.value()

            with open(self._settings_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

            self._lbl_save_hint.setText("✓ 配置已保存。请重启应用以使部分参数（如检测器路径）生效。")
            self.sig_config_saved.emit()

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"写入 settings.yaml 时出错：\n{e}")
