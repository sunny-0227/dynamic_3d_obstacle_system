"""
主界面状态文案与底栏摘要（纯展示逻辑）

从 MainWindow 拆出，避免主窗口同时承担布局与大量字符串拼装；
不涉及业务状态修改，仅根据 AppState + 控件当前选项生成展示文本。
"""

from __future__ import annotations

from typing import Any

from app.ui.controller import AppState
from app.ui.widgets.control_panel import ControlPanel
from app.ui.widgets.defense_status_bar import DefenseStatusBar


def mode_header_text(panel: ControlPanel, state: AppState, nusc_connected: bool) -> str:
    if state.workflow == "nuscenes" and nusc_connected:
        return "工作模式：nuScenes mini 数据集"
    if state.workflow == "single_file":
        return "工作模式：单文件点云"
    src = panel.ui_data_source()
    if src == "nuscenes":
        return "工作模式：nuScenes（尚未连接数据集）"
    return "工作模式：单文件点云（尚未选择文件）"


def build_summary_text(
    panel: ControlPanel,
    state: AppState,
    loader: Any,
    nusc_connected: bool,
) -> str:
    lines: list[str] = []
    lines.append(
        f"界面数据源：{'nuScenes mini' if panel.ui_data_source() == 'nuscenes' else '单文件点云'}"
    )
    wf_line = f"控制器工作流：{state.workflow}"
    if state.workflow == "none":
        wf_line += "（尚未选单文件或连接数据集）"
    lines.append(wf_line)

    if state.nusc_root:
        lines.append(f"数据集根目录：\n  {state.nusc_root.as_posix()}")
    else:
        lines.append("数据集根目录：（未选择）")

    if nusc_connected and loader is not None:
        lines.append(f"数据解析：{loader.mode_display_zh()}")
        lines.append(f"导航方式：{loader.navigation_display_zh()}")
        n = loader.frame_count
        idx = panel.frame_index()
        if n > 0:
            try:
                rec = loader.get_frame_record(idx)
                lines.append(f"当前场景：{rec.scene_name}")
                lines.append(f"当前帧索引：{idx} / {n - 1}（共 {n} 帧）")
                lines.append(f"当前帧 LiDAR 文件（磁盘）：\n  {rec.lidar_path.as_posix()}")
            except Exception:
                lines.append(f"当前帧索引：{idx}（元数据解析失败）")
        else:
            lines.append("当前场景 / 帧：（无可用帧）")
    else:
        lines.append("导航方式：（连接数据集后显示）")
        lines.append("当前场景 / 帧：（未连接数据集）")

    if state.current_file and state.workflow != "nuscenes":
        lines.append(f"单文件路径：\n  {state.current_file.as_posix()}")
    elif state.workflow == "nuscenes" and state.current_file:
        lines.append(f"已载入点云来源文件：\n  {state.current_file.as_posix()}")
    elif state.workflow == "nuscenes":
        lines.append("已载入点云来源文件：（尚未点击「加载当前帧点云」）")
    else:
        lines.append("单文件点云：（未选择或未载入）")

    if state.loaded_pcd is not None and len(state.loaded_pcd.points) > 0:
        lines.append(f"点云状态：已载入，{len(state.loaded_pcd.points):,} 点")
    else:
        lines.append("点云状态：未载入或为空")

    return "\n".join(lines)


def apply_defense_status_bar(
    bar: DefenseStatusBar,
    state: AppState,
    loader: Any,
    nusc_connected: bool,
) -> None:
    if nusc_connected and loader is not None:
        bar.set_dataset_status(
            f"数据集：已连接 ｜ {loader.mode_display_zh()} ｜ {loader.navigation_display_zh()}"
        )
    elif state.nusc_root:
        bar.set_dataset_status("数据集：已选根目录，待加载")
    else:
        bar.set_dataset_status("数据集：未选择")

    if state.loaded_pcd is not None and len(state.loaded_pcd.points) > 0:
        bar.set_frame_status(f"当前帧/点云：已载入 ｜ {len(state.loaded_pcd.points):,} 点")
    elif state.workflow == "nuscenes" and nusc_connected:
        bar.set_frame_status("当前帧：未载入 LiDAR 点云")
    elif state.workflow == "single_file":
        bar.set_frame_status("当前帧：单文件模式 — 请先加载点云")
    else:
        bar.set_frame_status("当前帧：—")
