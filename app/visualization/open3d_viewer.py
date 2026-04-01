"""
Open3D 可视化模块
采用弹出独立 Open3D 窗口的方式展示点云、语义分割结果和3D检测框。
不强行嵌入 PyQt5，避免跨平台 GUI 兼容性问题。
"""

from typing import List, Optional

import open3d as o3d

from app.core.fusion import FusionResult
from app.utils.logger import get_logger

logger = get_logger("visualization.open3d_viewer")


def show_pointcloud(
    pcd: o3d.geometry.PointCloud,
    window_title: str = "点云预览",
    width: int = 1280,
    height: int = 720,
    background_color: List[float] = None,
    point_size: float = 2.0,
) -> None:
    """
    在 Open3D 独立窗口中显示单个点云（无分割和检测结果）。

    参数：
        pcd:              要显示的点云
        window_title:     窗口标题
        width/height:     窗口尺寸（像素）
        background_color: 背景颜色 [R, G, B]（归一化）
        point_size:       点渲染大小
    """
    if background_color is None:
        background_color = [0.05, 0.05, 0.05]

    if len(pcd.points) == 0:
        logger.warning("点云为空，跳过显示")
        return

    logger.info("打开 Open3D 窗口显示点云，点数: %d", len(pcd.points))

    # 若点云没有颜色，统一赋为浅灰色
    if not pcd.has_colors():
        pcd_display = o3d.geometry.PointCloud(pcd)
        pcd_display.paint_uniform_color([0.7, 0.7, 0.7])
    else:
        pcd_display = pcd

    vis = _create_visualizer(window_title, width, height, background_color, point_size)
    vis.add_geometry(pcd_display)
    _run_visualizer(vis)


def show_fusion_result(
    fusion_result: FusionResult,
    window_title: str = "3D点云感知结果",
    width: int = 1280,
    height: int = 720,
    background_color: List[float] = None,
    point_size: float = 2.0,
) -> None:
    """
    在 Open3D 独立窗口中同时显示：
      - 语义分割着色点云
      - 3D 检测框线框

    参数：
        fusion_result:    FusionResult 融合结果对象
        window_title:     窗口标题
        width/height:     窗口尺寸（像素）
        background_color: 背景颜色 [R, G, B]（归一化）
        point_size:       点渲染大小
    """
    if background_color is None:
        background_color = [0.05, 0.05, 0.05]

    geometries = fusion_result.get_all_geometries()

    if not geometries:
        logger.warning("无可显示的几何体，跳过")
        return

    logger.info(
        "打开 Open3D 窗口，共 %d 个几何体（含 %d 个检测框）",
        len(geometries),
        len(fusion_result.detections),
    )

    # 打印检测结果摘要
    for i, det in enumerate(fusion_result.detections):
        logger.info(
            "  检测框 #%d | 类别: %-12s | 置信度: %.3f",
            i + 1, det.label, det.score,
        )

    vis = _create_visualizer(window_title, width, height, background_color, point_size)

    for geom in geometries:
        vis.add_geometry(geom)

    _run_visualizer(vis)


def _create_visualizer(
    window_title: str,
    width: int,
    height: int,
    background_color: List[float],
    point_size: float,
) -> o3d.visualization.Visualizer:
    """
    创建并配置 Open3D Visualizer 实例。
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=window_title,
        width=width,
        height=height,
    )

    # 设置渲染选项
    render_opt = vis.get_render_option()
    render_opt.background_color = background_color
    render_opt.point_size = point_size
    render_opt.show_coordinate_frame = True  # 显示坐标轴辅助参考

    return vis


def _run_visualizer(vis: o3d.visualization.Visualizer) -> None:
    """
    重置视角并进入交互式渲染循环，窗口关闭后销毁资源。
    """
    # 自动适配视角到场景范围
    vis.reset_view_point(True)
    logger.info("Open3D 窗口已打开，关闭窗口以继续操作")
    vis.run()
    vis.destroy_window()
    logger.info("Open3D 窗口已关闭")
