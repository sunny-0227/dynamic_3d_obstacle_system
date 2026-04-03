"""
检测框与分割结果的统一坐标融合（里程碑 5）

核心目标：
  1) 明确系统中的坐标系
  2) 若模块输出坐标一致：统一封装
  3) 若存在坐标变换：提供 Transform 并在此处完成对齐
  4) 输出可直接用于 Open3D 同窗显示的场景数据

当前工程默认（占位实现）：
  - 原始点云坐标系：raw_points（输入点云/算法使用点云的 xyz）
  - 检测框坐标系：与 raw_points 一致（center/size/yaw 基于同一坐标系）
  - 分割结果坐标系：与 raw_points 一致（labels 与点一一对应）
  - 可视化坐标系：Open3D 视窗坐标系，直接使用 raw_points

后续接入真实 nuScenes / 多传感器时：
  - 可以在这里把 detector/segmentor 的输出变换到统一坐标系（例如 LiDAR_TOP）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.core.detector.base_detector import DetectionBox
from app.core.geometry.transform import Transform, apply_transform_to_detection_box
from app.core.pipeline.segment_pipeline import SegmentPipelineOutput
from app.core.postprocess.box_converter import BoxConverter
from app.core.segmentor.base_segmentor import SegmentationResult
from app.utils.logger import get_logger

logger = get_logger("core.fusion.result_fusion")


@dataclass(frozen=True)
class CoordinateSystemSpec:
    """用于在日志/GUI 中明确坐标系含义。"""

    raw_points: str = "raw_points"
    detection_boxes: str = "raw_points"
    segmentation_labels: str = "raw_points"
    visualization: str = "raw_points(Open3D)"


@dataclass(frozen=True)
class FusedScene:
    """
    融合后可视化场景数据（统一坐标系）。

    points_xyz: (N,3) float32
    colored_pcd: Open3D PointCloud（可选）
    detections: 标准 DetectionBox 列表（统一坐标）
    det_obbs: Open3D OrientedBoundingBox 列表（可选，方便渲染）
    """

    coord: CoordinateSystemSpec
    points_xyz: np.ndarray
    seg: SegmentPipelineOutput
    detections: List[DetectionBox]
    det_obbs: Optional[list] = None


class ResultFusion:
    """
    负责：
      - 明确坐标系
      - 必要时对齐 detection/segmentation 到同一坐标系
      - 生成可渲染的 Open3D OBB 列表
    """

    def __init__(self, box_converter: Optional[BoxConverter] = None):
        self._box_converter = box_converter or BoxConverter()

    def fuse(
        self,
        points_xyz: np.ndarray,
        seg_out: SegmentPipelineOutput,
        detections: List[DetectionBox],
        T_det_to_points: Optional[Transform] = None,
        T_seg_to_points: Optional[Transform] = None,
        coord_spec: Optional[CoordinateSystemSpec] = None,
    ) -> FusedScene:
        """
        将分割与检测对齐到 points_xyz 的坐标系。
        - 若变换为 None：视为坐标一致
        - seg_out 当前默认与点云同坐标；若未来 seg 输出在别的坐标系，可在此对点/labels的对应关系做更复杂映射
        """
        coord = coord_spec or CoordinateSystemSpec()
        pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)

        det_std = self._box_converter.convert(detections)
        if T_det_to_points is not None:
            det_aligned: List[DetectionBox] = []
            for d in det_std:
                c2, s2, y2 = apply_transform_to_detection_box(d.center, d.size, d.yaw, T_det_to_points)
                det_aligned.append(
                    DetectionBox(
                        class_name=d.class_name,
                        score=float(d.score),
                        center=c2,
                        size=s2,
                        yaw=float(y2),
                    )
                )
            det_std = det_aligned
            logger.info("已对齐检测框坐标：det -> points")

        if T_seg_to_points is not None:
            # 重要：逐点 labels 与点索引耦合，真实模型若重采样/过滤点，需要显式 indices 映射
            logger.warning(
                "当前 SegmentPipelineOutput 的 labels 默认与输入点一一对应；"
                "若你设置 T_seg_to_points，说明 seg 与 points 坐标不同，"
                "但仍需保证点索引对齐（后续可扩展 indices 映射）。"
            )

        det_obbs = None
        try:
            det_obbs = self._box_converter.convert_to_open3d_obbs(det_std)
        except Exception as e:
            logger.warning("转换 Open3D OBB 失败（不影响数据输出）：%s", e)

        return FusedScene(
            coord=coord,
            points_xyz=pts,
            seg=seg_out,
            detections=det_std,
            det_obbs=det_obbs,
        )


def fuse_partial_for_gui_display(
    points_xyz: np.ndarray,
    last_seg: Optional[SegmentPipelineOutput],
    last_det: Optional[List[DetectionBox]],
    fusion: Optional[ResultFusion] = None,
) -> tuple[FusedScene, str, List[str]]:
    """
    主界面「融合显示」：在仅有部分算法结果时，用全零背景分割补齐并与检测框融合。

    与 MainWindow 中融合显示拼装逻辑一致，便于 UI 层只负责弹窗与 Open3D 调度。
    返回：(融合场景, 主日志行, 附加日志行列表)。
    """
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    has_seg = last_seg is not None
    has_det = bool(last_det) if last_det is not None else False

    if has_seg:
        seg_out = last_seg
    else:
        labels = np.zeros((pts.shape[0],), dtype=np.int32)
        seg = SegmentationResult(
            labels=labels,
            id_to_name={0: "background"},
            id_to_color={0: [0.7, 0.7, 0.7]},
        )
        seg_out = SegmentPipelineOutput(points_xyz=pts, seg=seg, colored_pcd=None)

    detections = list(last_det or [])
    rf = fusion or ResultFusion()
    scene = rf.fuse(points_xyz=pts, seg_out=seg_out, detections=detections)

    src = (
        "检测+分割拼装"
        if (has_det and has_seg)
        else "仅检测"
        if has_det
        else "仅分割"
        if has_seg
        else "空结果拼装"
    )
    primary = f"融合显示：{src}（点数={pts.shape[0]:,} | 检测框={len(detections)}）"
    extra: List[str] = []
    if has_seg:
        labels = seg_out.seg.labels
        uniq, cnt = np.unique(labels, return_counts=True) if labels.size > 0 else ([], [])
        extra.append("分割统计（前6类）")
        for lid, c in list(zip(list(uniq), list(cnt)))[:6]:
            name = seg_out.seg.id_to_name.get(int(lid), f"class_{int(lid)}")
            extra.append(f"  - id={int(lid):2d} | {name:12s} | 点数={int(c):,}")
    return scene, primary, extra

