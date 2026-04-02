"""
场景渲染器（里程碑 5）
在同一个 Open3D 视窗中同时显示：
  - 点云（原始 or 彩色分割结果）
  - 检测框（OrientedBoundingBox）

设计目标：
  - 与 GUI 解耦：不嵌入 PyQt5
  - 可复用：可直接渲染 FusedScene
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
            pcd.points = o3d.utility.Vector3dVector(scene.points_xyz.astype(np.float64, copy=False))
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
        vis.create_window(window_name=opts.window_title, width=opts.width, height=opts.height)
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

