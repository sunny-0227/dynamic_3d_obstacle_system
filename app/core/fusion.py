"""
结果融合模块
将伪检测结果与伪分割结果打包为统一的 FusionResult 数据结构，
供可视化模块统一调用。
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import open3d as o3d

from app.core.fake_detector import DetectionResult, run_fake_detection
from app.core.fake_segmentor import run_fake_segmentation
from app.utils.logger import get_logger

logger = get_logger("core.fusion")


@dataclass
class FusionResult:
    """
    融合结果数据类，包含原始点云、分割着色点云和3D检测框列表。
    """
    # 原始点云（未着色）
    raw_pcd: o3d.geometry.PointCloud
    # 语义分割后的着色点云
    segmented_pcd: o3d.geometry.PointCloud
    # 每个点的语义类别标签，shape (N,)
    seg_labels: np.ndarray
    # 3D 检测框结果列表
    detections: List[DetectionResult] = field(default_factory=list)

    def get_all_geometries(self) -> List[o3d.geometry.Geometry3D]:
        """
        返回所有可用于 Open3D 可视化的几何体列表。
        包含：分割着色点云 + 所有检测框线框。
        """
        geometries: List[o3d.geometry.Geometry3D] = [self.segmented_pcd]
        for det in self.detections:
            if det.box_geometry is not None:
                geometries.append(det.box_geometry)
        return geometries


def run_full_pipeline(
    pcd: o3d.geometry.PointCloud,
    num_boxes: int = 3,
    score_range: tuple = (0.75, 0.99),
    num_classes: int = 4,
) -> FusionResult:
    """
    执行完整的伪推理流水线：语义分割 + 3D目标检测，并将结果融合。

    参数：
        pcd:          原始输入点云
        num_boxes:    伪检测框数量
        score_range:  置信度随机区间
        num_classes:  语义类别数量

    返回：
        FusionResult 数据对象
    """
    logger.info("开始完整推理流水线")

    # 步骤1：执行伪语义分割
    logger.info("步骤1/2：语义分割")
    segmented_pcd, seg_labels = run_fake_segmentation(
        pcd, num_classes=num_classes
    )

    # 步骤2：执行伪3D目标检测
    logger.info("步骤2/2：3D目标检测")
    detections = run_fake_detection(
        pcd, num_boxes=num_boxes, score_range=score_range
    )

    # 打包融合结果
    result = FusionResult(
        raw_pcd=pcd,
        segmented_pcd=segmented_pcd,
        seg_labels=seg_labels,
        detections=detections,
    )

    logger.info(
        "流水线完成 | 分割点数: %d | 检测框数: %d",
        len(seg_labels),
        len(detections),
    )
    return result
