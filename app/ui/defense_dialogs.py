"""
答辩版常用 QMessageBox 封装（文案与按钮组合与原先 MainWindow 完全一致）。
仅做结构整理，不改变任何用户可见提示。
"""

from __future__ import annotations

from PyQt5.QtWidgets import QMessageBox, QWidget


def show_error_critical(parent: QWidget, title: str, message: str) -> None:
    """控制器 sig_error 等：严重错误弹窗。"""
    QMessageBox.critical(parent, title, message)


def info_need_single_data_source_for_load(parent: QWidget) -> None:
    QMessageBox.information(
        parent,
        "操作提示",
        "当前未选择「单文件点云」数据源。\n请在左侧 ① 中选中「单文件点云」后再加载。",
    )


def info_need_nusc_data_source_for_connect(parent: QWidget) -> None:
    QMessageBox.information(
        parent,
        "操作提示",
        "当前未选择「nuScenes mini」数据源。\n请在左侧 ① 中选中该项后再加载数据集。",
    )


def info_need_single_data_source_for_pick_file(parent: QWidget) -> None:
    QMessageBox.information(
        parent,
        "操作提示",
        "请先选择「单文件点云」数据源，再选择点云文件。",
    )


def info_need_nusc_data_source_for_root(parent: QWidget) -> None:
    QMessageBox.information(
        parent,
        "操作提示",
        "请先选择「nuScenes mini」数据源，再选择数据集根目录。",
    )


def ask_switch_data_source_to_nuscenes_clears_single(parent: QWidget) -> bool:
    r = QMessageBox.question(
        parent,
        "切换数据源",
        "将切换到 nuScenes 模式，已选单文件路径会被清空（未保存结果也会清空）。\n\n是否继续？",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return r == QMessageBox.Yes


def ask_switch_data_source_to_single_disconnects_nusc(parent: QWidget) -> bool:
    r = QMessageBox.question(
        parent,
        "切换数据源",
        "将切换到单文件模式，nuScenes 数据集连接会断开。\n\n是否继续？",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return r == QMessageBox.Yes


def ask_pick_file_disconnects_nusc(parent: QWidget) -> bool:
    r = QMessageBox.question(
        parent,
        "切换工作流",
        "当前已连接 nuScenes 数据集。选择单文件将断开数据集连接。\n\n是否继续？",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return r == QMessageBox.Yes


def ask_nusc_root_clears_single_file(parent: QWidget) -> bool:
    r = QMessageBox.question(
        parent,
        "切换工作流",
        "当前为单文件模式。选择 nuScenes 根目录将清空已选单文件路径。\n\n是否继续？",
        QMessageBox.Yes | QMessageBox.No,
        QMessageBox.No,
    )
    return r == QMessageBox.Yes


def warn_need_nonempty_pcd_for_fusion(parent: QWidget) -> None:
    QMessageBox.warning(parent, "提示", "请先加载非空点云后再使用融合显示。")


def warn_empty_pcd_cannot_preview(parent: QWidget) -> None:
    QMessageBox.warning(parent, "提示", "当前点云为空，无法打开 Open3D 预览。")
