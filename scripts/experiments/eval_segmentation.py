"""
scripts/experiments/eval_segmentation.py

几何分割统计与 IoU 实验模块。

当前系统使用 RANSAC 地面分割 + DBSCAN 聚类，无人工逐点标注。
因此不输出严格语义 IoU，只输出几何规则统计指标，并在 csv 中备注说明。

输出字段：
  file_name, point_count, ground_ratio, nonground_ratio,
  cluster_count, obstacle_count, pseudo_iou, coverage_ratio,
  label_type, note
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 点云读取（复用项目已有逻辑，失败则使用内置简化版）
# ---------------------------------------------------------------------------

def load_point_cloud(file_path: Path) -> Optional[np.ndarray]:
    """
    读取 .bin 或 .pcd 点云文件，返回 (N, 3) 或 (N, 4) float32 数组。
    失败返回 None。
    """
    try:
        if file_path.suffix.lower() == ".bin":
            pts = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, 4)
            return pts[:, :3]
        elif file_path.suffix.lower() in (".pcd", ".ply"):
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(file_path))
                return np.asarray(pcd.points, dtype=np.float32)
            except ImportError:
                pass
            # open3d 不可用时尝试手动解析 ASCII PCD
            return _parse_pcd_ascii(file_path)
        else:
            print(f"[eval_seg] 不支持的格式: {file_path.suffix}")
            return None
    except Exception as e:
        print(f"[eval_seg] 读取失败 {file_path}: {e}")
        return None


def _parse_pcd_ascii(file_path: Path) -> Optional[np.ndarray]:
    """简单解析 ASCII 格式 PCD，只取 x y z。"""
    try:
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().upper() == "DATA ascii":
                data_start = i + 1
                break
        pts = []
        for line in lines[data_start:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        return np.array(pts, dtype=np.float32) if pts else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RANSAC 地面分割（优先复用项目代码，降级为内置实现）
# ---------------------------------------------------------------------------

def _ransac_ground(
    pts: np.ndarray,
    distance_threshold: float = 0.20,
    iterations: int = 50,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    RANSAC 平面拟合地面分割。

    返回：
        ground_mask   (N,) bool，True = 地面点
        nonground_pts (M, 3) 非地面点
        plane_normal  (3,) 平面法向量，若拟合失败为 None
    """
    N = len(pts)
    best_inliers = np.zeros(N, dtype=bool)
    best_count   = 0
    best_normal  = None

    for _ in range(iterations):
        idx = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue
        normal = normal / norm_len
        d = -np.dot(normal, p0)
        dists = np.abs(pts @ normal + d)
        inliers = dists < distance_threshold
        count = np.sum(inliers)
        if count > best_count:
            best_count   = count
            best_inliers = inliers
            best_normal  = normal

    nonground_pts = pts[~best_inliers]
    return best_inliers, nonground_pts, best_normal


def ransac_ground_segment(
    pts: np.ndarray,
    distance_threshold: float = 0.20,
    iterations: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    调用项目内 LightweightSegmentor 或降级到内置 RANSAC。
    返回 (ground_mask, nonground_pts)。
    """
    # 尝试复用项目已有分割器
    try:
        # 将项目根加入 sys.path
        _add_project_root()
        from app.realtime.realtime_segmentor import LightweightSegmentor, GroundSegConfig
        cfg = GroundSegConfig(
            distance_threshold_m=distance_threshold,
            ransac_iterations=iterations,
        )
        seg = LightweightSegmentor(cfg)
        result = seg.segment(pts)
        # LightweightSegmentor 返回 SegmentOutput，有 labels 字段
        labels = result.labels  # (N,) int
        ground_mask = labels == 1  # SEG_GROUND = 1
        nonground_pts = pts[~ground_mask]
        return ground_mask, nonground_pts
    except Exception:
        pass

    # 降级：内置 RANSAC
    ground_mask, nonground_pts, _ = _ransac_ground(pts, distance_threshold, iterations)
    return ground_mask, nonground_pts


# ---------------------------------------------------------------------------
# DBSCAN 聚类
# ---------------------------------------------------------------------------

def dbscan_cluster(
    pts: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 10,
) -> np.ndarray:
    """
    对点云做 DBSCAN 聚类。
    返回 labels 数组 (N,)，-1 表示噪声。
    优先使用 scikit-learn，降级到简化版。
    """
    if len(pts) == 0:
        return np.array([], dtype=int)
    try:
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree", n_jobs=1)
        return db.fit_predict(pts)
    except ImportError:
        pass

    # 降级：简单 KD-Tree DBSCAN
    return _simple_dbscan(pts, eps, min_samples)


def _simple_dbscan(pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """极简 DBSCAN（仅用于 sklearn 不可用时降级），性能较低。"""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        neighbors = tree.query_ball_point(pts, r=eps)
        N = len(pts)
        labels = np.full(N, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(N, dtype=bool)
        for i in range(N):
            if visited[i]:
                continue
            visited[i] = True
            nb = neighbors[i]
            if len(nb) < min_samples:
                continue
            labels[i] = cluster_id
            stack = list(nb)
            while stack:
                j = stack.pop()
                if not visited[j]:
                    visited[j] = True
                    nb2 = neighbors[j]
                    if len(nb2) >= min_samples:
                        stack.extend(nb2)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
        return labels
    except Exception:
        return np.zeros(len(pts), dtype=int)


# ---------------------------------------------------------------------------
# 伪 IoU 计算（无真值标注时的稳定性近似）
# ---------------------------------------------------------------------------

def compute_pseudo_iou(
    ground_mask: np.ndarray,
    pts: np.ndarray,
    ransac_threshold: float = 0.20,
    n_trials: int = 5,
) -> float:
    """
    通过多次独立 RANSAC 分割，用不同随机种子的结果交并比近似稳定性。
    结果越接近 1.0 表示地面分割越稳定。
    注意：这不是严格语义 IoU，仅为几何稳定性指标。
    """
    if len(pts) < 100:
        return 0.0

    masks = []
    for seed in range(n_trials):
        np.random.seed(seed)
        gm, _, _ = _ransac_ground(pts, distance_threshold=ransac_threshold, iterations=30)
        masks.append(gm)

    # 用第一次与后续的平均 IoU 近似整体稳定性
    ref = masks[0].astype(float)
    ious = []
    for m in masks[1:]:
        m = m.astype(float)
        intersection = np.sum(ref * m)
        union = np.sum(np.clip(ref + m, 0, 1))
        if union > 0:
            ious.append(intersection / union)
    return round(float(np.mean(ious)), 4) if ious else 0.0


# ---------------------------------------------------------------------------
# 单文件评估
# ---------------------------------------------------------------------------

def eval_one_file(
    file_path: Path,
    ransac_threshold: float = 0.20,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 10,
    cluster_min_points: int = 15,
) -> Dict:
    """
    对单个点云文件执行几何分割统计，返回指标字典。
    """
    pts = load_point_cloud(file_path)
    if pts is None or len(pts) == 0:
        return {
            "file_name":       file_path.name,
            "point_count":     0,
            "ground_ratio":    None,
            "nonground_ratio": None,
            "cluster_count":   None,
            "obstacle_count":  None,
            "pseudo_iou":      None,
            "coverage_ratio":  None,
            "label_type":      "none",
            "note":            "文件读取失败或为空",
        }

    N = len(pts)

    # 地面分割
    ground_mask, nonground_pts = ransac_ground_segment(pts, ransac_threshold)
    ground_count    = int(np.sum(ground_mask))
    nonground_count = N - ground_count
    ground_ratio    = round(ground_count / N, 4)
    nonground_ratio = round(nonground_count / N, 4)

    # DBSCAN 聚类（仅对非地面点）
    cluster_count   = 0
    obstacle_count  = 0
    coverage_ratio  = 0.0

    if nonground_count > 0:
        labels = dbscan_cluster(nonground_pts, eps=dbscan_eps, min_samples=dbscan_min_samples)
        unique_labels = set(labels) - {-1}
        cluster_count = len(unique_labels)

        # 满足最小点数的视为障碍物候选
        obstacle_count = sum(
            1 for cid in unique_labels
            if int(np.sum(labels == cid)) >= cluster_min_points
        )

        # coverage_ratio：非地面中被聚类覆盖的点比例（不含噪声点 -1）
        covered = int(np.sum(labels >= 0))
        coverage_ratio = round(covered / nonground_count, 4) if nonground_count > 0 else 0.0

    # 伪 IoU（地面分割稳定性）
    pseudo_iou = compute_pseudo_iou(ground_mask, pts, ransac_threshold)

    return {
        "file_name":       file_path.name,
        "point_count":     N,
        "ground_ratio":    ground_ratio,
        "nonground_ratio": nonground_ratio,
        "cluster_count":   cluster_count,
        "obstacle_count":  obstacle_count,
        "pseudo_iou":      pseudo_iou,
        "coverage_ratio":  coverage_ratio,
        "label_type":      "geometric_rule",
        "note":            "无人工逐点标签，该结果为几何规则统计，不作为严格语义 IoU",
    }


# ---------------------------------------------------------------------------
# 路径工具（内部使用）
# ---------------------------------------------------------------------------

def _add_project_root():
    """将项目根目录加入 sys.path，使项目模块可被导入。"""
    # 脚本位于 scripts/experiments/，项目根在上两级
    root = Path(__file__).resolve().parent.parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    """
    扫描 data_dir 下所有 .bin / .pcd 文件，逐一执行几何分割统计。
    输出 segmentation_iou.csv。
    """
    data_dir = Path(args.data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集点云文件
    files: List[Path] = []
    for ext in ("*.bin", "*.pcd", "*.ply"):
        files.extend(sorted(data_dir.rglob(ext)))

    if not files:
        print(f"[eval_seg] ⚠ 未在 {data_dir} 找到点云文件（.bin / .pcd / .ply）")
        df = pd.DataFrame(columns=[
            "file_name", "point_count", "ground_ratio", "nonground_ratio",
            "cluster_count", "obstacle_count", "pseudo_iou", "coverage_ratio",
            "label_type", "note",
        ])
        csv_path = output_dir / "segmentation_iou.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[eval_seg] 已保存（空表）: {csv_path}")
        return df

    print(f"[eval_seg] 共找到 {len(files)} 个点云文件，开始评估...")
    rows = []

    for fp in files:
        print(f"[eval_seg]   处理: {fp.name}")
        try:
            row = eval_one_file(
                fp,
                ransac_threshold=getattr(args, "ransac_threshold", 0.20),
                dbscan_eps=getattr(args, "dbscan_eps", 0.5),
                dbscan_min_samples=getattr(args, "dbscan_min_samples", 10),
            )
            rows.append(row)
        except Exception as e:
            print(f"[eval_seg] ✗ {fp.name} 处理异常: {e}")
            rows.append({
                "file_name": fp.name, "point_count": 0,
                "ground_ratio": None, "nonground_ratio": None,
                "cluster_count": None, "obstacle_count": None,
                "pseudo_iou": None, "coverage_ratio": None,
                "label_type": "error",
                "note": f"处理异常: {e}",
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "segmentation_iou.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[eval_seg] 已保存: {csv_path}（共 {len(df)} 条）")
    return df


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="几何分割统计评估模块")
    parser.add_argument("--data_dir",            default="data")
    parser.add_argument("--output_dir",          default="outputs/experiments")
    parser.add_argument("--ransac_threshold",    type=float, default=0.20)
    parser.add_argument("--dbscan_eps",          type=float, default=0.50)
    parser.add_argument("--dbscan_min_samples",  type=int,   default=10)
    args = parser.parse_args()

    _add_project_root()
    out = Path(args.output_dir)
    main(args, out)
