"""
场景渲染器：在 Open3D 窗口中同时显示点云（原始/分割彩色）和检测框。

两种渲染模式：
  - render(scene)          — 阻塞式，离线一次性显示
  - open_realtime_window() + update(scene) + tick() + close()  — 非阻塞实时渲染
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import open3d as o3d

from app.core.fusion.result_fusion import FusedScene
from app.utils.logger import get_logger

logger = get_logger("visualization.scene_renderer")


@dataclass(frozen=True)
class RenderOptions:
    window_title: str = "融合显示（点云+分割+检测框）"
    width: int = 1280
    height: int = 720
    background_color: Optional[List[float]] = None
    point_size: float = 2.0
    show_coordinate_frame: bool = True


class SceneRenderer:
    def __init__(self, opts: Optional[RenderOptions] = None):
        self._opts = opts or RenderOptions()
        # 非阻塞实时窗口状态
        self._vis: Optional[o3d.visualization.Visualizer] = None
        self._rt_pcd: Optional[o3d.geometry.PointCloud] = None
        self._rt_obbs: List[o3d.geometry.OrientedBoundingBox] = []
        self._first_frame: bool = True  # 第一帧需要 reset_view_point

    # ------------------------------------------------------------------
    # ① 阻塞式渲染（离线模式，行为与原来完全一致）
    # ------------------------------------------------------------------

    def render(self, scene: FusedScene) -> None:
        """
        打开 Open3D 窗口并显示场景。
        注意：该调用会阻塞直到用户关闭窗口。
        """
        opts = self._opts
        bg = opts.background_color or [0.05, 0.05, 0.05]

        geometries: List[o3d.geometry.Geometry3D] = []

        # 优先显示分割后的彩色点云；否则显示原始点云（灰色）
        if scene.seg.colored_pcd is not None:
            pcd = scene.seg.colored_pcd
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                scene.points_xyz.astype(np.float64, copy=False)
            )
            pcd.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(pcd)

        if scene.det_obbs:
            for obb in scene.det_obbs:
                geometries.append(obb)

        if not geometries:
            logger.warning("无可渲染几何体，跳过显示")
            return

        logger.info(
            "渲染场景 | 点数=%d | 检测框=%d | 坐标系：points=%s det=%s seg=%s vis=%s",
            scene.points_xyz.shape[0],
            len(scene.detections),
            scene.coord.raw_points,
            scene.coord.detection_boxes,
            scene.coord.segmentation_labels,
            scene.coord.visualization,
        )

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=opts.window_title, width=opts.width, height=opts.height
        )
        render_opt = vis.get_render_option()
        render_opt.background_color = np.asarray(bg, dtype=np.float64)
        render_opt.point_size = float(opts.point_size)
        render_opt.show_coordinate_frame = bool(opts.show_coordinate_frame)

        for g in geometries:
            vis.add_geometry(g)

        vis.reset_view_point(True)
        logger.info("Open3D 窗口已打开，关闭窗口以继续操作")
        vis.run()
        vis.destroy_window()
        logger.info("Open3D 窗口已关闭")

    # ------------------------------------------------------------------
    # ② 非阻塞持久化窗口（实时模式）
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """返回实时窗口当前是否已打开。"""
        return self._vis is not None

    def open_realtime_window(self) -> None:
        """
        创建并打开非阻塞 Open3D 实时窗口。

        调用后立即返回，不阻塞 GUI 主线程。
        需要在调用方的定时器里持续调用 tick() 驱动 Open3D 事件循环。
        """
        if self._vis is not None:
            logger.debug("[SceneRenderer] 实时窗口已存在，跳过重复创建")
            return

        opts = self._opts
        bg = opts.background_color or [0.05, 0.05, 0.05]

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=opts.window_title,
            width=opts.width,
            height=opts.height,
        )
        render_opt = vis.get_render_option()
        render_opt.background_color = np.asarray(bg, dtype=np.float64)
        render_opt.point_size = float(opts.point_size)
        render_opt.show_coordinate_frame = bool(opts.show_coordinate_frame)

        # 预先加入一个空点云占位（方便后续直接 update_geometry）
        placeholder = o3d.geometry.PointCloud()
        vis.add_geometry(placeholder)
        self._rt_pcd = placeholder
        self._rt_obbs = []
        self._first_frame = True
        self._vis = vis
        logger.info("[SceneRenderer] 实时窗口已打开：%s", opts.window_title)

    def update(self, scene: FusedScene) -> bool:
        """
        用新的 FusedScene 刷新实时窗口（非阻塞）。

        返回 False 表示窗口已被用户关闭，调用方应停止实时分析。
        """
        if self._vis is None:
            return False

        # 检测窗口是否已被用户手动关闭
        if not self._vis.poll_events():
            logger.info("[SceneRenderer] 实时窗口被用户关闭")
            self._cleanup()
            return False

        # ── 更新点云 ──────────────────────────────────────────────
        new_pcd: o3d.geometry.PointCloud
        if scene.seg.colored_pcd is not None:
            new_pcd = scene.seg.colored_pcd
        else:
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(
                scene.points_xyz.astype(np.float64, copy=False)
            )
            new_pcd.paint_uniform_color([0.7, 0.7, 0.7])

        # 将新点云数据复制进已注册的占位 PointCloud，避免重新 add/remove_geometry
        self._rt_pcd.points = new_pcd.points
        self._rt_pcd.colors = new_pcd.colors
        self._vis.update_geometry(self._rt_pcd)

        # ── 更新边界框：先移除旧的，再添加新的 ─────────────────────
        for old_obb in self._rt_obbs:
            try:
                self._vis.remove_geometry(old_obb, reset_bounding_box=False)
            except Exception:
                pass
        self._rt_obbs = []

        new_obbs = scene.det_obbs or []
        for obb in new_obbs:
            self._vis.add_geometry(obb, reset_bounding_box=False)
            self._rt_obbs.append(obb)

        # 第一帧重置视角，之后保持用户当前视角
        if self._first_frame:
            self._vis.reset_view_point(True)
            self._first_frame = False

        self._vis.update_renderer()
        return True

    def tick(self) -> bool:
        """
        驱动 Open3D 非阻塞事件循环（在 GUI 定时器里调用）。

        返回 False 表示窗口已关闭，调用方可停止定时器。
        """
        if self._vis is None:
            return False
        if not self._vis.poll_events():
            logger.info("[SceneRenderer] 实时窗口被用户关闭（tick 检测）")
            self._cleanup()
            return False
        self._vis.update_renderer()
        return True

    def close(self) -> None:
        """主动关闭并销毁实时窗口（幂等）。"""
        if self._vis is not None:
            logger.info("[SceneRenderer] 主动关闭实时窗口")
            self._cleanup()

    def _cleanup(self) -> None:
        """销毁 Open3D 窗口并重置状态。"""
        if self._vis is not None:
            try:
                self._vis.destroy_window()
            except Exception:
                pass
            self._vis = None
        self._rt_pcd = None
        self._rt_obbs = []
        self._first_frame = True

