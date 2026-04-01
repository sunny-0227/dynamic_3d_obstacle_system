"""
伪语义分割模块（第一阶段占位实现）
不接真实模型，对点云中每个点随机分配语义类别，并根据类别着色。
着色结果直接写入 Open3D PointCloud 的 colors 属性。
"""

from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d

from app.utils.logger import get_logger

logger = get_logger("core.fake_segmentor")

# 默认语义类别及对应颜色（归一化 RGB）
DEFAULT_CLASS_COLORS: Dict[int, List[float]] = {
    0: [0.5, 0.5, 0.5],   # 背景 - 灰色
    1: [0.2, 0.8, 0.2],   # 植被 - 绿色
    2: [0.8, 0.2, 0.2],   # 障碍物 - 红色
    3: [0.2, 0.4, 0.9],   # 地面 - 蓝色
}

DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "背景",
    1: "植被",
    2: "障碍物",
    3: "地面",
}


def run_fake_segmentation(
    pcd: o3d.geometry.PointCloud,
    num_classes: int = 4,
    class_colors: Dict[int, List[float]] | None = None,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    对输入点云执行伪语义分割，返回着色后的点云副本及类别标签数组。

    分割策略（伪实现）：
    - 按点的 Z 轴高度做粗略分层，最低层归为"地面"（类别3）
    - 其余点随机分配剩余类别，加入少量噪声使视觉效果更真实

    参数：
        pcd:          输入点云（不修改原始对象）
        num_classes:  语义类别数量
        class_colors: 类别颜色字典，默认使用 DEFAULT_CLASS_COLORS

    返回：
        (着色后的 PointCloud 副本, shape(N,) 的类别标签 ndarray)
    """
    logger.info("开始伪语义分割，类别数: %d", num_classes)

    if class_colors is None:
        class_colors = DEFAULT_CLASS_COLORS

    points = np.asarray(pcd.points)
    n_points = len(points)

    if n_points == 0:
        logger.warning("点云为空，跳过语义分割")
        colored_pcd = o3d.geometry.PointCloud(pcd)
        return colored_pcd, np.array([], dtype=np.int32)

    # 初始化所有点的类别为背景（0）
    labels = np.zeros(n_points, dtype=np.int32)

    # 按 Z 高度阈值划分"地面"层（最低 20% 的点归为地面）
    z_values = points[:, 2]
    z_threshold = np.percentile(z_values, 20)
    ground_mask = z_values <= z_threshold
    labels[ground_mask] = 3  # 地面

    # 非地面点随机分配其余类别（0, 1, 2）
    non_ground_idx = np.where(~ground_mask)[0]
    if len(non_ground_idx) > 0:
        random_labels = np.random.randint(0, 3, size=len(non_ground_idx))
        labels[non_ground_idx] = random_labels

    # 统计各类别数量并记录日志
    for cls_id in range(num_classes):
        count = np.sum(labels == cls_id)
        cls_name = DEFAULT_CLASS_NAMES.get(cls_id, f"类别{cls_id}")
        logger.debug("类别 %d（%s）: %d 个点", cls_id, cls_name, count)

    # 根据类别标签生成颜色数组
    colors = np.zeros((n_points, 3), dtype=np.float64)
    for cls_id, color in class_colors.items():
        mask = labels == cls_id
        colors[mask] = color

    # 创建点云副本并写入颜色
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(points.copy())
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)

    logger.info("伪语义分割完成，共处理 %d 个点", n_points)
    return colored_pcd, labels
