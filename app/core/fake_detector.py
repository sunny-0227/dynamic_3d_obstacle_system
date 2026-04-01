"""
伪3D目标检测模块（第一阶段占位实现）
不接真实模型，根据输入点云的空间范围随机生成 2~3 个 3D 检测框。
每个检测框用 open3d.geometry.OrientedBoundingBox 表示，并附带类别和置信度。
"""

import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import open3d as o3d

from app.utils.logger import get_logger

logger = get_logger("core.fake_detector")

# 伪类别标签
FAKE_CLASSES = ["car", "pedestrian", "cyclist"]

# 各类别对应的伪框颜色 [R, G, B]（归一化 0~1）
CLASS_COLORS = {
    "car":        [1.0, 0.3, 0.0],   # 橙红
    "pedestrian": [1.0, 1.0, 0.0],   # 黄色
    "cyclist":    [0.0, 1.0, 1.0],   # 青色
}

# 各类别典型尺寸 [长, 宽, 高]（单位：米）
CLASS_SIZES = {
    "car":        [4.5, 2.0, 1.6],
    "pedestrian": [0.8, 0.8, 1.8],
    "cyclist":    [1.8, 0.8, 1.6],
}


@dataclass
class DetectionResult:
    """单个3D检测结果数据类"""
    label: str                          # 类别名称
    score: float                        # 置信度 [0, 1]
    center: np.ndarray                  # 框中心坐标 (3,)
    size: np.ndarray                    # 框尺寸 [长, 宽, 高]
    rotation_y: float                   # 绕 Y 轴旋转角（弧度）
    box_geometry: o3d.geometry.OrientedBoundingBox = field(default=None, repr=False)


def run_fake_detection(
    pcd: o3d.geometry.PointCloud,
    num_boxes: int = 3,
    score_range: tuple = (0.75, 0.99),
) -> List[DetectionResult]:
    """
    对输入点云执行伪3D检测，返回检测结果列表。

    参数：
        pcd:         输入点云
        num_boxes:   伪造检测框数量（2~3 个）
        score_range: 置信度随机区间

    返回：
        List[DetectionResult]，每个元素包含类别、置信度、几何体等信息
    """
    logger.info("开始伪检测，目标框数量: %d", num_boxes)

    points = np.asarray(pcd.points)

    if len(points) == 0:
        logger.warning("点云为空，跳过检测")
        return []

    # 计算点云的 XY 平面范围，用于随机放置检测框
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()
    z_base = points[:, 2].min()  # 地面高度近似值

    results: List[DetectionResult] = []
    num_boxes = max(2, min(num_boxes, 3))  # 限制在 2~3 个

    for i in range(num_boxes):
        # 随机选择类别
        label = random.choice(FAKE_CLASSES)
        score = round(random.uniform(*score_range), 3)

        # 在点云 XY 范围内随机选取中心点
        cx = random.uniform(x_min + 1.0, x_max - 1.0)
        cy = random.uniform(y_min + 1.0, y_max - 1.0)
        size = CLASS_SIZES[label]
        # 中心高度为地面 + 半个框高
        cz = z_base + size[2] / 2.0

        center = np.array([cx, cy, cz])

        # 随机绕 Y 轴旋转角度
        rotation_y = random.uniform(0, np.pi)

        # 构建 Open3D 有向包围框
        box = _build_obb(center, size, rotation_y, CLASS_COLORS[label])

        result = DetectionResult(
            label=label,
            score=score,
            center=center,
            size=np.array(size),
            rotation_y=rotation_y,
            box_geometry=box,
        )
        results.append(result)

        logger.debug(
            "检测框 #%d | 类别: %s | 置信度: %.3f | 中心: (%.2f, %.2f, %.2f)",
            i + 1, label, score, cx, cy, cz,
        )

    logger.info("伪检测完成，共生成 %d 个检测框", len(results))
    return results


def _build_obb(
    center: np.ndarray,
    size: list,
    rotation_y: float,
    color: list,
) -> o3d.geometry.OrientedBoundingBox:
    """
    根据中心点、尺寸和旋转角构建 Open3D OrientedBoundingBox。

    参数：
        center:     框中心坐标 (3,)
        size:       [长, 宽, 高]
        rotation_y: 绕 Y 轴旋转弧度（KITTI 坐标系）
        color:      线框颜色 [R, G, B]

    返回：
        open3d.geometry.OrientedBoundingBox
    """
    # Open3D 使用 Z 轴向上的右手坐标系，KITTI 绕 Y 轴旋转对应 Open3D 绕 Z 轴旋转
    # 此处做简化处理，直接在 XY 平面内旋转
    cos_r = np.cos(rotation_y)
    sin_r = np.sin(rotation_y)

    # 旋转矩阵（绕 Z 轴，即俯视平面内旋转）
    R = np.array([
        [cos_r, -sin_r, 0.0],
        [sin_r,  cos_r, 0.0],
        [0.0,    0.0,   1.0],
    ])

    extent = np.array(size, dtype=np.float64)  # [长, 宽, 高]

    obb = o3d.geometry.OrientedBoundingBox(
        center=center.astype(np.float64),
        R=R,
        extent=extent,
    )
    obb.color = color
    return obb
