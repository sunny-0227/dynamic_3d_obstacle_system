from __future__ import annotations

"""
实时轻量障碍物检测模块

两层职责：
  1. LightweightDetector（核心算法）
       - 接受「障碍物候选点云」（来自 LightweightSegmentor 的 SEG_OBSTACLE 点）
       - 使用 DBSCAN 聚类，将连续点云分割为独立障碍物实例
       - 为每个有效聚类生成 3D 边界框（AABB 轴对齐或近似 OBB）
       - 过滤过小/过大聚类（噪点或墙体）
       - 返回标准 DetectionBox 列表 + Open3D OBB 列表（可直接渲染）

  2. RealtimeDetector（流水线适配层）
       - 保留旧版「复用离线 DetectPipeline」接口不变
       - 新增「轻量几何模式」供 LightweightRealtimePipeline 调用

算法原理：
  DBSCAN（Density-Based Spatial Clustering of Applications with Noise）：
    - 核心参数：epsilon（邻域半径）、min_samples（核心点最小邻点数）
    - 优点：不需要预设簇数；能识别任意形状的聚类；自动标记离群噪点为 -1
    - 复杂度：O(N log N)（KD-Tree 加速），典型 3D 点云 (~5k 障碍物点) < 10ms

  边界框生成：
    方案 A（AABB，默认）：
      各轴 min/max → center + half-extent → DetectionBox
      优点：极快（< 0.1ms/框），缺点：不含旋转角 yaw
    方案 B（PCA-OBB）：
      对各簇点做 PCA，第一主成分方向为 yaw → 更紧凑的有向边界框
      优点：贴合细长障碍物；缺点：约 1ms/框

  类别推断（简单规则）：
    高 h > 1.5m, 面积 < 2m²  → "person"（行人）
    高 h > 1.2m, 面积 > 1m²  → "car"
    其余                      → "obstacle"

不依赖：
  - 深度学习模型 / GPU
  - PCL
  - open3d（仅 OBB 可视化生成时用，可选）
  - scikit-learn（DBSCAN 优先用 scipy KD-Tree 自实现，若不可用降级）
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.core.detector.base_detector import DetectionBox
from app.core.pipeline.detect_pipeline import DetectPipeline
from app.utils.logger import get_logger

logger = get_logger("realtime.detector")


# ---------------------------------------------------------------------------
# DBSCAN 聚类配置
# ---------------------------------------------------------------------------
@dataclass
class ClusterConfig:
    """DBSCAN 聚类参数配置。"""

    # DBSCAN epsilon：邻域半径（米）
    # 典型室外点云 0.5~1.0m；RealSense 室内 0.15~0.3m
    epsilon_m: float = 0.5
    # DBSCAN 核心点最小邻点数（含自身）
    min_samples: int = 10
    # 聚类最小点数：小于此值视为噪点聚类，不生成框
    cluster_min_points: int = 15
    # 聚类最大点数：大于此值视为背景/墙体，不生成框
    cluster_max_points: int = 20000
    # 生成框的最小体积（m³）：过滤极小聚类（地面残留点等）
    box_min_volume_m3: float = 0.002
    # 生成框的最大对角线长（m）：过滤过大聚类（整面墙）
    box_max_diagonal_m: float = 15.0
    # 是否使用 PCA 计算有向边界框（OBB）；否则使用 AABB（更快）
    use_pca_obb: bool = False
    # 类别推断：行人高度下限（米）
    person_min_height_m: float = 1.2
    # 类别推断：车辆面积下限（m²，xy 平面投影面积）
    car_min_area_m2: float = 1.0


# ---------------------------------------------------------------------------
# 聚类结果数据结构（内部使用）
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    """单个聚类的几何结果。"""

    # 聚类内点在原始点云中的索引
    indices: np.ndarray           # (K,) int
    # 点云中心（质心）
    centroid: np.ndarray          # (3,) float32
    # AABB 或 OBB center
    center: np.ndarray            # (3,) float32
    # 尺寸 [l, w, h]（l=length=x 方向，w=width=y 方向，h=height=z 方向）
    size: np.ndarray              # (3,) float32
    # yaw（绕 Z 轴旋转角，弧度）；AABB 时为 0
    yaw: float
    # 点数
    n_points: int


# ---------------------------------------------------------------------------
# 轻量几何检测器（核心）
# ---------------------------------------------------------------------------
class LightweightDetector:
    """
    基于 DBSCAN 聚类的轻量 3D 障碍物检测器。

    输入：障碍物候选点云（通常来自 LightweightSegmentor 的 SEG_OBSTACLE 点）
    输出：List[DetectionBox]  +  Open3D OBB 列表（可选）
    """

    def __init__(self, config: Optional[ClusterConfig] = None) -> None:
        self._cfg = config or ClusterConfig()

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def detect(
        self,
        obstacle_points: np.ndarray,
    ) -> Tuple[List[DetectionBox], List[ClusterResult]]:
        """
        对障碍物候选点执行聚类检测。

        参数：
          obstacle_points — (M, 3) float32，仅包含障碍物类别的点

        返回：
          detections    — 标准 DetectionBox 列表
          clusters      — ClusterResult 列表（含索引，可映射回原始点云）
        """
        pts = np.asarray(obstacle_points, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        pts_xyz = pts[:, :3]
        M = pts_xyz.shape[0]

        if M < self._cfg.cluster_min_points:
            logger.debug("[LightweightDetector] 障碍物候选点过少（%d），跳过聚类", M)
            return [], []

        # 步骤 1：DBSCAN 聚类
        labels_cluster = self._dbscan(pts_xyz)

        # 步骤 2：提取有效聚类，生成边界框
        unique_labels = np.unique(labels_cluster)
        clusters: List[ClusterResult] = []
        detections: List[DetectionBox] = []

        for lbl in unique_labels:
            if lbl == -1:
                continue  # DBSCAN 噪点
            mask = labels_cluster == lbl
            n = int(mask.sum())

            cfg = self._cfg
            # 过小聚类：噪点残留，直接跳过
            if n < cfg.cluster_min_points:
                continue
            # 过大聚类：通常是墙体/地面残留，跳过而非下采样（避免产生误框）
            if n > cfg.cluster_max_points:
                logger.debug(
                    "[LightweightDetector] 跳过过大聚类（%d 点 > max %d）", n, cfg.cluster_max_points
                )
                continue
            cluster_pts = pts_xyz[mask]

            # 生成边界框
            if cfg.use_pca_obb:
                cr = self._compute_pca_obb(cluster_pts, np.where(mask)[0])
            else:
                cr = self._compute_aabb(cluster_pts, np.where(mask)[0])

            if cr is None:
                continue

            # 过滤过小/过大的框
            vol = float(np.prod(cr.size))
            diag = float(np.linalg.norm(cr.size))
            if vol < cfg.box_min_volume_m3 or diag > cfg.box_max_diagonal_m:
                continue

            # 类别推断
            class_name = self._infer_class(cr)

            det = DetectionBox(
                class_name=class_name,
                score=1.0,  # 几何方法无置信度，统一填 1.0
                center=cr.center.copy(),
                size=cr.size.copy(),
                yaw=float(cr.yaw),
            )
            clusters.append(cr)
            detections.append(det)

        logger.debug(
            "[LightweightDetector] 聚类完成 | 障碍物点数：%d | 有效聚类：%d",
            M,
            len(detections),
        )
        return detections, clusters

    def detect_with_obbs(
        self,
        obstacle_points: np.ndarray,
    ) -> Tuple[List[DetectionBox], List[ClusterResult], Optional[list]]:
        """
        检测 + 生成 Open3D OBB 列表（可直接传入 SceneRenderer）。

        返回：(detections, clusters, open3d_obbs | None)
        """
        detections, clusters = self.detect(obstacle_points)
        obbs = self._detections_to_open3d_obbs(detections)
        return detections, clusters, obbs

    # ------------------------------------------------------------------
    # DBSCAN 实现
    # ------------------------------------------------------------------

    def _dbscan(self, pts: np.ndarray) -> np.ndarray:
        """
        执行 DBSCAN 聚类。

        优先使用 sklearn（速度最快），其次 scipy 自实现，最后简单 grid 降级。
        返回：(N,) int 标签数组，-1 表示噪点。
        """
        cfg = self._cfg
        # 优先路径：sklearn
        try:
            from sklearn.cluster import DBSCAN
            db = DBSCAN(
                eps=cfg.epsilon_m,
                min_samples=cfg.min_samples,
                algorithm="kd_tree",
                n_jobs=1,
            ).fit(pts)
            return db.labels_.astype(np.int32)
        except ImportError:
            pass

        # 次选路径：scipy KD-Tree 手动实现
        try:
            return self._dbscan_scipy(pts)
        except ImportError:
            pass

        # 降级路径：体素网格聚类（无 scipy/sklearn 时）
        return self._voxel_cluster(pts)

    def _dbscan_scipy(self, pts: np.ndarray) -> np.ndarray:
        """
        使用 scipy cKDTree 实现标准 DBSCAN 的 BFS 扩展逻辑。

        标准 DBSCAN 三个步骤：
          1. 对每个未访问点，查询 epsilon 邻域
          2. 若邻域点数 >= min_samples，该点为核心点，开始新聚类
          3. 递归扩展：将核心点的所有邻域点（未分配的）加入聚类
        """
        from scipy.spatial import cKDTree

        cfg = self._cfg
        N = pts.shape[0]
        tree = cKDTree(pts)

        # 一次性批量计算所有点的邻域（比逐点查询快 30~50%）
        neighborhoods = tree.query_ball_point(pts, r=cfg.epsilon_m)

        labels = np.full(N, -1, dtype=np.int32)
        # visited 单独维护：区分「已访问但是噪点」和「未访问」
        visited = np.zeros(N, dtype=bool)
        cluster_id = 0

        for i in range(N):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = neighborhoods[i]
            if len(neighbors) < cfg.min_samples:
                # 非核心点，暂标为噪点（-1），后续若被其他核心点扩展则会重新赋值
                continue

            # i 是核心点，开启新聚类
            labels[i] = cluster_id
            # 用 set 维护待扩展队列，天然去重，避免重复入队
            seed_set = set(neighbors)
            seed_set.discard(i)  # i 已处理

            while seed_set:
                q = seed_set.pop()
                if not visited[q]:
                    visited[q] = True
                    q_neighbors = neighborhoods[q]
                    if len(q_neighbors) >= cfg.min_samples:
                        # q 也是核心点，将其邻域中未分配的点加入待扩展集
                        for nb in q_neighbors:
                            if labels[nb] == -1:
                                seed_set.add(nb)
                # 无论 q 是否已访问，只要未被分配就归入当前聚类
                if labels[q] == -1:
                    labels[q] = cluster_id

            cluster_id += 1

        return labels

    def _voxel_cluster(self, pts: np.ndarray) -> np.ndarray:
        """
        降级方案：体素网格聚类（无外部依赖）。

        将空间划分为 voxel，同一 voxel 及其 3×3×3 邻域内的点属于同一聚类。
        精度低于 DBSCAN，但在无法导入 scipy/sklearn 时保证基本功能。
        """
        cfg = self._cfg
        vox_size = cfg.epsilon_m
        # 将点映射到体素索引
        vox_idx = np.floor(pts / vox_size).astype(np.int32)
        # 用 dict 建立体素→点索引的映射
        vox_map: dict = {}
        for i, vi in enumerate(map(tuple, vox_idx)):
            vox_map.setdefault(vi, []).append(i)

        labels = np.full(pts.shape[0], -1, dtype=np.int32)
        cluster_id = 0

        offsets = [(dx, dy, dz) for dx in range(-1, 2)
                   for dy in range(-1, 2)
                   for dz in range(-1, 2)]

        for vox, pt_ids in vox_map.items():
            # 注意：此处 min_samples 用于判断「体素是否足够稠密」，
            # 语义与 DBSCAN 的「核心点邻域点数」不同，调参时需注意
            if len(pt_ids) < cfg.min_samples:
                continue  # 稀疏体素跳过
            if labels[pt_ids[0]] != -1:
                continue  # 已分配
            # BFS 扩展连通体素
            queue = [vox]
            visited = {vox}
            while queue:
                cv = queue.pop()
                for pt_i in vox_map.get(cv, []):
                    if labels[pt_i] == -1:
                        labels[pt_i] = cluster_id
                for dx, dy, dz in offsets:
                    nv = (cv[0] + dx, cv[1] + dy, cv[2] + dz)
                    if nv not in visited and nv in vox_map:
                        if len(vox_map[nv]) >= cfg.min_samples:
                            visited.add(nv)
                            queue.append(nv)
            cluster_id += 1

        return labels

    # ------------------------------------------------------------------
    # 边界框计算
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_aabb(
        pts: np.ndarray, indices: np.ndarray
    ) -> Optional[ClusterResult]:
        """计算轴对齐边界框（AABB）。"""
        if pts.shape[0] < 3:
            return None
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        size = mx - mn
        if np.any(size < 1e-6):
            return None
        center = ((mn + mx) / 2.0).astype(np.float32)
        centroid = pts.mean(axis=0).astype(np.float32)
        return ClusterResult(
            indices=indices,
            centroid=centroid,
            center=center,
            size=size.astype(np.float32),
            yaw=0.0,
            n_points=int(pts.shape[0]),
        )

    @staticmethod
    def _compute_pca_obb(
        pts: np.ndarray, indices: np.ndarray
    ) -> Optional[ClusterResult]:
        """
        PCA 有向边界框（OBB）。

        用点集的 XY 平面协方差矩阵主方向作为 yaw（不对 Z 轴做旋转），
        在主方向坐标系下计算 min/max 得到紧凑边界框。
        """
        if pts.shape[0] < 4:
            return None

        pts_f64 = pts.astype(np.float64)
        centroid = pts_f64.mean(axis=0)
        pts_c = pts_f64 - centroid

        # 只用 XY 分量做 PCA，Z 轴独立处理
        xy = pts_c[:, :2]
        cov = (xy.T @ xy) / max(1, pts.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 主方向（最大特征值对应的向量）
        main_axis = eigvecs[:, np.argmax(eigvals)]
        yaw = float(np.arctan2(main_axis[1], main_axis[0]))

        # 旋转到主方向坐标系
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float64)
        xy_rot = (rot @ xy.T).T
        pts_rot = np.column_stack([xy_rot, pts_c[:, 2]])

        mn = pts_rot.min(axis=0)
        mx = pts_rot.max(axis=0)
        size = (mx - mn).astype(np.float32)
        if np.any(size < 1e-6):
            return None

        # 在旋转坐标系里的中心，转回原始坐标系
        center_rot = (mn + mx) / 2.0
        cos_back, sin_back = np.cos(yaw), np.sin(yaw)
        rot_back = np.array([[cos_back, -sin_back], [sin_back, cos_back]], dtype=np.float64)
        center_xy = rot_back @ center_rot[:2] + centroid[:2]
        center_z = center_rot[2] + centroid[2]
        center = np.array([center_xy[0], center_xy[1], center_z], dtype=np.float32)

        return ClusterResult(
            indices=indices,
            centroid=centroid.astype(np.float32),
            center=center,
            size=size,
            yaw=yaw,
            n_points=int(pts.shape[0]),
        )

    # ------------------------------------------------------------------
    # 类别推断
    # ------------------------------------------------------------------

    def _infer_class(self, cr: ClusterResult) -> str:
        """
        根据聚类几何特征推断障碍物类别（简单规则）。

        这是一个极简启发式分类，适合演示；接入真实模型后可替换此函数。
        """
        cfg = self._cfg
        h = float(cr.size[2])                   # Z 方向高度
        area = float(cr.size[0] * cr.size[1])   # XY 投影面积

        if h >= cfg.person_min_height_m and area < cfg.car_min_area_m2:
            return "person"
        if h >= 0.5 and area >= cfg.car_min_area_m2:
            return "car"
        return "obstacle"

    # ------------------------------------------------------------------
    # Open3D OBB 生成（可选）
    # ------------------------------------------------------------------

    @staticmethod
    def _detections_to_open3d_obbs(
        detections: List[DetectionBox],
    ) -> Optional[list]:
        """将 DetectionBox 列表转为 Open3D OrientedBoundingBox 列表。"""
        try:
            import open3d as o3d
        except ImportError:
            return None

        # 不同类别使用不同颜色
        color_map = {
            "car":      [0.9, 0.5, 0.1],   # 橙色
            "person":   [0.1, 0.8, 0.1],   # 绿色
            "obstacle": [0.9, 0.1, 0.1],   # 红色
        }
        obbs = []
        for d in detections:
            cos_r = float(np.cos(d.yaw))
            sin_r = float(np.sin(d.yaw))
            R = np.array(
                [[cos_r, -sin_r, 0.0],
                 [sin_r,  cos_r, 0.0],
                 [0.0,    0.0,   1.0]],
                dtype=np.float64,
            )
            obb = o3d.geometry.OrientedBoundingBox(
                center=d.center.astype(np.float64),
                R=R,
                extent=d.size.astype(np.float64),
            )
            obb.color = color_map.get(d.class_name, [0.9, 0.1, 0.1])
            obbs.append(obb)
        return obbs


# ---------------------------------------------------------------------------
# 适配层：保持旧版离线 DetectPipeline 接口不变
# ---------------------------------------------------------------------------
class RealtimeDetector:
    """
    实时检测适配层。

    两种构造模式：
      1. 传入 DetectPipeline    → 复用离线大模型（占位/真实均可）
      2. 传入 LightweightDetector → 使用轻量几何方法
      3. 不传参数              → 自动创建 LightweightDetector（默认配置）

    注意：run() 的输入在轻量模式下应为「障碍物候选点」（SEG_OBSTACLE 子集），
          而非全量点云，以减少不必要的聚类计算量。
    """

    def __init__(
        self,
        detect_pipeline: Optional[DetectPipeline] = None,
        lightweight: Optional[LightweightDetector] = None,
    ) -> None:
        self._pipeline = detect_pipeline
        self._light = lightweight
        if detect_pipeline is None and lightweight is None:
            self._light = LightweightDetector()

    def run(self, points_xyz: np.ndarray) -> List[DetectionBox]:
        """执行检测，返回 DetectionBox 列表。"""
        if self._pipeline is not None:
            return self._pipeline.run(points_xyz)
        detections, _ = self._light.detect(points_xyz)
        return detections

    def run_with_obbs(
        self, points_xyz: np.ndarray
    ) -> Tuple[List[DetectionBox], Optional[list]]:
        """执行检测并返回 Open3D OBB 列表（供直接渲染）。"""
        if self._pipeline is not None:
            dets = self._pipeline.run(points_xyz)
            return dets, None
        dets, _, obbs = self._light.detect_with_obbs(points_xyz)
        return dets, obbs
