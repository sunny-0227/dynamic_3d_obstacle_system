"""
主窗口辅助函数：AppState → OfflinePage 状态同步。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.ui.controller import AppState

if TYPE_CHECKING:
    from app.datasets.nuscenes_loader import NuScenesMiniLoader
    from app.ui.pages.offline_page import OfflinePage


def compute_action_button_flags(state: AppState) -> tuple[bool, bool, bool]:
    """
    返回 (has_nonempty_pcd, has_results, allow_run_full_autoload)，
    供 OfflinePage.set_action_buttons_state 使用。
    """
    has_nonempty = state.loaded_pcd is not None and len(state.loaded_pcd.points) > 0
    has_results = (
        state.last_det is not None
        or state.last_seg is not None
        or state.last_scene is not None
    )
    allow_autoload = (
        state.workflow != "nuscenes"
        and state.loaded_pcd is None
        and state.current_file is not None
        and state.current_file.is_file()
    )
    return has_nonempty, has_results, allow_autoload


def refresh_nusc_meta_line(page: "OfflinePage", loader: "NuScenesMiniLoader") -> None:
    """根据当前帧索引更新 nuScenes 元信息一行。"""
    if not loader.is_connected:
        return
    n = loader.frame_count
    if n <= 0:
        page.set_nusc_meta_text("当前导航下无帧")
        return
    idx = page.frame_index()
    try:
        rec = loader.get_frame_record(idx)
        page.set_nusc_meta_text(
            f"帧 {idx + 1}/{n} ｜ {rec.scene_name} ｜ {rec.lidar_path.name}"
        )
    except Exception:
        page.set_nusc_meta_text(f"帧 {idx + 1}/{n}（解析失败）")
