"""
实时轻量语义分割模块

算法：RANSAC 平面拟合地面分割 + 高度/密度过滤识别障碍物候选。
不依赖深度学习模型、GPU、PCL，纯 numpy 实现。

语义类别：0=背景, 1=地面, 2=障碍物候选, 3=孤立噪点
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.pipeline.segment_pipeline import SegmentPipeline, SegmentPipelineOutput
from app.core.segmentor.base_segmentor import SegmentationResult
from app.utils.logger import get_logger

logger = get_logger("realtime.segmentor")


# ---------------------------------------------------------------------------
# 语义类别常量（供外部模块引用，避免魔法数字）
# ---------------------------------------------------------------------------
SEG_BACKGROUND: int = 0
SEG_GROUND: int = 1
SEG_OBSTACLE: int = 2
SEG_NOISE: int = 3

# 各类别的 RGB 颜色（归一化 0~1）
_DEFAULT_ID_TO_COLOR: Dict[int, List[float]] = {
    SEG_BACKGROUND: [0.55, 0.55, 0.55],  # 灰色
    SEG_GROUND:     [0.65, 0.45, 0.20],  # 棕黄（地面感）
    SEG_OBSTACLE:   [0.90, 0.20, 0.20],  # 红色（障碍物告警感）
    SEG_NOISE:      [0.30, 0.30, 0.30],  # 深灰
}

_DEFAULT_ID_TO_NAME: Dict[int, str] = {
    SEG_BACKGROUND: "background",
    SEG_GROUND:     "ground",
    SEG_OBSTACLE:   "obstacle",
    SEG_NOISE:      "noise",
}


# ---------------------------------------------------------------------------
# RANSAC 地面拟合参数（dataclass，方便外部调整）
# ---------------------------------------------------------------------------
@dataclass
class GroundSegConfig:
    """RANSAC 地面分割配置参数。"""

    # RANSAC 迭代次数：越大越精确，但耗时线性增长
    ransac_iterations: int = 50
    # 内点判定阈值（米）：点到平面距离 < threshold 视为内点（地面点）
    distance_threshold_m: float = 0.20
    # 最小内点比例：地面点占总点数的比例低于此值时认为平面拟合失败
    min_inlier_ratio: float = 0.10
    # 非地面点中，高于地面法向距离多少米以上视为「障碍物候选」下限
    obstacle_min_height_m: float = 0.10
    # 非地面点中，高于地面法向距离超过此值视为「背景」（高处噪点/天花板）
    obstacle_max_height_m: float = 3.50
    # 若点云 Z 轴中位数落在此范围内，认为 Z 是高度轴（允许倒置相机）
    z_range_for_height: Tuple[float, float] = (-5.0, 5.0)
    # 噪点过滤：孤立点（邻近点数 < min_neighbors 则标记为 noise）
    # 0 = 不做噪点过滤（默认关闭，避免增加耗时）
    noise_min_neighbors: int = 0
    noise_radius_m: float = 0.5


# ---------------------------------------------------------------------------
# 轻量几何分割器（核心）
# ---------------------------------------------------------------------------
class LightweightSegmentor:
    """
    基于 RANSAC 平面拟合的轻量点云语义分割器。

    接口与 BaseSegmentor 兼容，但不继承它（避免不必要的约束）。
    外层通过 LightweightRealtimePipeline 统一调用。
    """

    def __init__(self, config: Optional[GroundSegConfig] = None) -> None:
        self._cfg = config or GroundSegConfig()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def segment(self, points_xyz: np.ndarray) -> SegmentationResult:
        """
        对 (N,3) 或 (N,K) float32 点云执行分割。

        返回：SegmentationResult
          labels    : (N,) int32，各点语义类别
          id_to_name: 类别 id → 名称
          id_to_color: 类别 id → [r, g, b]
        """
        pts = np.asarray(points_xyz, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        pts_xyz = pts[:, :3]
        N = pts_xyz.shape[0]

        labels = np.zeros(N, dtype=np.int32)  # 默认全部为背景

        if N < 10:
            logger.debug("[LightweightSegmentor] 点数过少（%d），跳过分割", N)
            return SegmentationResult(
                labels=labels,
                id_to_name=dict(_DEFAULT_ID_TO_NAME),
                id_to_color=dict(_DEFAULT_ID_TO_COLOR),
            )

        # 步骤 1：RANSAC 地面拟合
        ground_mask, plane_params = self._ransac_ground(pts_xyz)

        if ground_mask is not None:
            labels[ground_mask] = SEG_GROUND

            # 步骤 2：非地面点按高度分类
            non_ground_mask = ~ground_mask
            non_ground_idx = np.where(non_ground_mask)[0]

            if non_ground_idx.size > 0:
                # 计算每个非地面点到地面平面的有符号距离
                heights = self._signed_dist_to_plane(
                    pts_xyz[non_ground_idx], plane_params
                )

                cfg = self._cfg
                # 高度在 [min, max] 范围内 → 障碍物候选
                obstacle_sel = (heights >= cfg.obstacle_min_height_m) & (
                    heights <= cfg.obstacle_max_height_m
                )
                labels[non_ground_idx[obstacle_sel]] = SEG_OBSTACLE
                # 其余非地面点（高于 max 或低于 min，如负高度噪点）保持 background(0)

        # 步骤 3：（可选）噪点过滤
        if self._cfg.noise_min_neighbors > 0:
            labels = self._filter_noise(pts_xyz, labels)

        return SegmentationResult(
            labels=labels,
            id_to_name=dict(_DEFAULT_ID_TO_NAME),
            id_to_color=dict(_DEFAULT_ID_TO_COLOR),
        )

    def segment_with_colored_pcd(
        self, points_xyz: np.ndarray
    ) -> Tuple[SegmentationResult, Optional[object]]:
        """
        分割并生成 Open3D 彩色点云（若 open3d 不可用则 colored_pcd=None）。

        返回：(SegmentationResult, open3d.geometry.PointCloud | None)
        """
        seg_result = self.segment(points_xyz)
        colored_pcd = self._build_colored_pcd(points_xyz, seg_result)
        return seg_result, colored_pcd

    # ------------------------------------------------------------------
    # RANSAC 核心
    # ------------------------------------------------------------------

    def _ransac_ground(
        self, pts_xyz: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        RANSAC 平面拟合：返回地面内点 mask 和平面参数 [a,b,c,d]（ax+by+cz+d=0）。

        若拟合失败（内点不足）返回 (None, None)。
        """
        N = pts_xyz.shape[0]
        cfg = self._cfg
        best_mask = None
        best_count = 0
        min_inliers = max(10, int(N * cfg.min_inlier_ratio))

        # 预采样：优先从 Z 轴较低的 30% 点中取样，提高地面点抽到的概率
        z_vals = pts_xyz[:, 2]
        z_low_thresh = float(np.percentile(z_vals, 30))
        low_idx = np.where(z_vals <= z_low_thresh)[0]
        if low_idx.size < 3:
            low_idx = np.arange(N)

        rng = np.random.default_rng(seed=42)  # 固定种子保证每帧结果一致性

        for _ in range(cfg.ransac_iterations):
            # 随机取 3 点确定平面
            sample_idx = rng.choice(low_idx, size=3, replace=False)
            p0, p1, p2 = pts_xyz[sample_idx]

            # 平面法向量（叉积）
            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm_len = float(np.linalg.norm(normal))
            if norm_len < 1e-8:
                # 三点共线，退出本次迭代
                continue
            normal = normal / norm_len
            d = -float(np.dot(normal, p0))

            # 计算所有点到平面的绝对距离
            dists = np.abs(pts_xyz @ normal + d)

            # 统计内点
            inlier_mask = dists < cfg.distance_threshold_m
            cnt = int(inlier_mask.sum())
            if cnt > best_count:
                best_count = cnt
                best_mask = inlier_mask
                best_normal = normal
                best_d = d

        if best_count < min_inliers:
            logger.debug(
                "[LightweightSegmentor] RANSAC 失败，内点数 %d < 最小要求 %d",
                best_count,
                min_inliers,
            )
            return None, None

        plane_params = np.array(
            [best_normal[0], best_normal[1], best_normal[2], best_d], dtype=np.float32
        )
        logger.debug(
            "[LightweightSegmentor] RANSAC 地面拟合成功 | 内点 %d/%d (%.1f%%)",
            best_count,
            N,
            100.0 * best_count / N,
        )
        return best_mask, plane_params

    @staticmethod
    def _signed_dist_to_plane(
        pts: np.ndarray, plane: np.ndarray
    ) -> np.ndarray:
        """
        计算点集到平面 [a,b,c,d] 的有符号距离（正值=法向正方向一侧）。
        这里用于判断「高于地面多少米」，法向量指向上方时正值表示在地面上方。
        """
        normal = plane[:3]
        d = float(plane[3])
        # 确保法向量指向 Z 正方向（向上），若 z 分量为负则翻转
        if normal[2] < 0:
            normal = -normal
            d = -d
        return (pts @ normal + d).astype(np.float32)

    # ------------------------------------------------------------------
    # 噪点过滤（可选）
    # ------------------------------------------------------------------

    def _filter_noise(self, pts_xyz: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        半径邻域过滤：邻近点数不足 min_neighbors 的点标记为 noise。
        使用 KD-Tree 加速，仅对非地面点执行，避免过度过滤地面平面内的稀疏点。
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            logger.debug("[LightweightSegmentor] scipy 不可用，跳过噪点过滤")
            return labels

        cfg = self._cfg
        non_ground = np.where(labels != SEG_GROUND)[0]
        if non_ground.size == 0:
            return labels

        tree = cKDTree(pts_xyz[non_ground])
        # 查询每个点的邻近点数（含自身，故 > min_neighbors+1）
        counts = tree.query_ball_point(pts_xyz[non_ground], r=cfg.noise_radius_m, return_length=True)
        noise_sel = counts <= cfg.noise_min_neighbors
        labels_copy = labels.copy()
        labels_copy[non_ground[noise_sel]] = SEG_NOISE
        return labels_copy

    # ------------------------------------------------------------------
    # 彩色点云生成
    # ------------------------------------------------------------------

    @staticmethod
    def _build_colored_pcd(
        points_xyz: np.ndarray, seg_result: SegmentationResult
    ) -> Optional[object]:
        """将分割结果映射到点云颜色，返回 Open3D PointCloud。"""
        try:
            import open3d as o3d
        except ImportError:
            return None

        pts = np.asarray(points_xyz, dtype=np.float64)[:, :3]
        N = pts.shape[0]
        colors = np.zeros((N, 3), dtype=np.float64)
        for label_id, color in seg_result.id_to_color.items():
            mask = seg_result.labels == label_id
            if mask.any():
                colors[mask] = np.array(color, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd


# ---------------------------------------------------------------------------
# 适配层：包装旧版离线 SegmentPipeline（保持向后兼容）
# ---------------------------------------------------------------------------
class RealtimeSegmentor:
    """
    实时分割适配层。

    两种构造模式：
      1. 传入 SegmentPipeline   → 复用离线大模型（占位/真实均可）
      2. 传入 LightweightSegmentor → 使用轻量几何方法
      3. 不传参数              → 自动创建 LightweightSegmentor（默认配置）

    外层 pipeline 通过 run() 调用，返回 SegmentPipelineOutput。
    """

    def __init__(
        self,
        segment_pipeline: Optional[SegmentPipeline] = None,
        lightweight: Optional[LightweightSegmentor] = None,
    ) -> None:
        self._pipeline = segment_pipeline
        self._light = lightweight
        if segment_pipeline is None and lightweight is None:
            self._light = LightweightSegmentor()

    def run(self, points_xyz: np.ndarray) -> SegmentPipelineOutput:
        """
        执行分割，返回与 SegmentPipeline 一致的 SegmentPipelineOutput。
        """
        pts_xyz = np.array(points_xyz[:, :3], dtype=np.float32)

        if self._pipeline is not None:
            return self._pipeline.run(pts_xyz)

        # 轻量几何模式
        seg_result, colored_pcd = self._light.segment_with_colored_pcd(pts_xyz)
        return SegmentPipelineOutput(
            points_xyz=pts_xyz,
            seg=seg_result,
            colored_pcd=colored_pcd,
        )
