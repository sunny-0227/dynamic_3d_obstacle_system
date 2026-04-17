"""
文件/目录选择（QFileDialog），默认目录与过滤器与主窗口约定一致。
使用 pathlib.Path，兼容 Windows 路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QWidget


def pick_pointcloud_file(
    parent: QWidget,
    project_root: Path,
    pointcloud_config: dict,
) -> Optional[Path]:
    """
    打开「选择点云文件」对话框。
    返回所选路径；用户取消则返回 None。
    """
    default_dir = str(
        project_root / pointcloud_config.get("default_data_dir", "data")
    )
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择点云文件",
        default_dir,
        "点云文件 (*.bin *.pcd);;所有文件 (*.*)",
    )
    return Path(file_path) if file_path else None


def pick_nuscenes_root_directory(parent: QWidget, project_root: Path) -> Optional[Path]:
    """选择 nuScenes mini 数据集根目录；取消返回 None。"""
    default_dir = str(project_root)
    picked = QFileDialog.getExistingDirectory(
        parent, "选择 nuScenes mini 数据集根目录", default_dir
    )
    return Path(picked) if picked else None


def pick_realtime_stream_directory(parent: QWidget, project_root: Path) -> Optional[Path]:
    """选择实时 Mock 点云流目录；取消返回 None。"""
    default_dir = str(project_root / "data")
    picked = QFileDialog.getExistingDirectory(parent, "选择实时点云流目录（Mock）", default_dir)
    return Path(picked) if picked else None
