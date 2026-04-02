"""
检测结果后处理与格式转换（里程碑 3）

OpenPCDet / MMDetection3D 在输出格式上可能存在差异；
本模块提供统一的 BoxConverter：
  - 将任意模型输出转换到统一的 DetectionBox 标准格式
  - 当前阶段：默认接收已经标准化的 DetectionBox（占位实现为主）
  - 后续接入真实模型：在这里替换实现 pred_boxes -> center/size/yaw
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from app.core.detector.base_detector import DetectionBox
from app.utils.logger import get_logger

logger = get_logger("core.postprocess.box_converter")


class BoxConverter:
    """
    将 detector 输出标准化为 List[DetectionBox]。
    """

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        self.class_names = class_names

    def convert(self, detections: List[DetectionBox]) -> List[DetectionBox]:
        """
        当前阶段：detections 已经是 DetectionBox，做轻量校验并返回。
        """
        if not detections:
            return []

        std: List[DetectionBox] = []
        for d in detections:
            center = np.asarray(d.center, dtype=np.float32).reshape(-1)
            size = np.asarray(d.size, dtype=np.float32).reshape(-1)
            if center.shape[0] != 3 or size.shape[0] != 3:
                raise ValueError(
                    f"DetectionBox 维度不正确：center={center.shape}, size={size.shape}"
                )
            std.append(
                DetectionBox(
                    class_name=str(d.class_name),
                    score=float(d.score),
                    center=center,
                    size=size,
                    yaw=float(d.yaw),
                )
            )
        return std

    def convert_openpcdet(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
    ) -> List[DetectionBox]:
        """
        OpenPCDet 真实输出到统一标准的转换接口（后续替换）。

        约定：
          - pred_boxes 通常为 shape (M, 7)
          - 格式约为 [x, y, z, l, w, h, rotation_y]
          - rotation_y 作为 yaw 输出
        """
        raise NotImplementedError("后续接入真实 OpenPCDet 时在这里实现转换。")

    def convert_to_open3d_obbs(
        self,
        detections: List[DetectionBox],
        color: Optional[List[float]] = None,
    ):
        """
        将标准 DetectionBox 转为 Open3D OrientedBoundingBox 列表，供可视化模块使用。

        注意：
            - Open3D 的 OBB 旋转矩阵使用右手系；此处将 yaw 视为绕 Z 轴旋转
            - 该转换不依赖 nuScenes/OpenPCDet，纯几何转换
        """
        import open3d as o3d

        if color is None:
            color = [1.0, 0.3, 0.0]  # 默认橙红

        obbs = []
        for d in self.convert(detections):
            cos_r = float(np.cos(d.yaw))
            sin_r = float(np.sin(d.yaw))
            R = np.array(
                [
                    [cos_r, -sin_r, 0.0],
                    [sin_r, cos_r, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            obb = o3d.geometry.OrientedBoundingBox(
                center=d.center.astype(np.float64),
                R=R,
                extent=d.size.astype(np.float64),
            )
            obb.color = color
            obbs.append(obb)

        return obbs

