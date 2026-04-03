"""
将 AppState 同步到左侧 ControlPanel（与原先 MainWindow._on_state 内面板部分一致）。

主窗口仍负责顶栏/摘要/底栏与控制器；此处只处理「左侧面板控件与 workflow 对齐」。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.ui.controller import AppState
from app.ui.widgets.control_panel import ControlPanel

if TYPE_CHECKING:
    from app.datasets.nuscenes_loader import NuScenesMiniLoader


def compute_action_button_flags(state: AppState) -> tuple[bool, bool, bool]:
    """
    返回 (has_nonempty_pcd, has_results, allow_run_full_autoload)，
    供 set_action_buttons_state 使用，逻辑与原先 _on_state 中一致。
    """
    has_nonempty = state.loaded_pcd is not None and len(state.loaded_pcd.points) > 0
    has_results = (
        (state.last_det is not None)
        or (state.last_seg is not None)
        or (state.last_scene is not None)
    )
    allow_autoload = (
        state.workflow != "nuscenes"
        and state.loaded_pcd is None
        and state.current_file is not None
        and state.current_file.is_file()
    )
    return has_nonempty, has_results, allow_autoload


def refresh_nusc_meta_line(panel: ControlPanel, loader: NuScenesMiniLoader) -> None:
    """根据当前帧索引更新 nuScenes 元信息一行（原 _update_nusc_meta 主体）。"""
    if not loader.is_connected:
        return
    n = loader.frame_count
    if n <= 0:
        panel.set_nusc_meta_text("当前导航下无帧")
        return
    idx = panel.frame_index()
    try:
        rec = loader.get_frame_record(idx)
        panel.set_nusc_meta_text(
            f"帧 {idx+1}/{n} ｜ {rec.scene_name} ｜ {rec.lidar_path.name}"
        )
    except Exception:
        panel.set_nusc_meta_text(f"帧 {idx+1}/{n}（解析失败）")


def sync_control_panel_to_state(panel: ControlPanel, state: AppState) -> None:
    """根据 state 刷新数据源单选锁定、nuScenes 列表/导航、路径显示等。"""
    loader = state.nusc_loader
    nusc_connected = loader is not None and loader.is_connected

    if state.workflow == "nuscenes":
        panel.sync_data_source_radio("nuscenes")
    elif state.workflow == "single_file":
        panel.sync_data_source_radio("single_file")

    panel.apply_source_module_lock(panel.ui_data_source())

    if nusc_connected:
        scenes = loader.get_scene_summaries()
        items = [(s["name"], s["token"]) for s in scenes]
        panel.set_scene_list(items)
        panel.set_nusc_nav_enabled(True, frame_count=loader.frame_count)
        refresh_nusc_meta_line(panel, loader)
    else:
        panel.set_nusc_nav_enabled(False, frame_count=0)
        panel.set_nusc_meta_text("请先选择根目录并加载数据集")

    panel.set_selected_file(state.current_file)
    panel.set_nusc_root(state.nusc_root)
