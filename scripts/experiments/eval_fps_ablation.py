"""
scripts/experiments/eval_fps_ablation.py

FPS 对比实验 + 参数敏感性消融实验模块。

一、FPS 对比实验：
  测试不同点云规模（原始 / 50000 / 20000 / 10000 点）下，
  RANSAC + DBSCAN 完整处理链的运行耗时与 FPS。
  每种规模重复 5 次取均值。

二、参数敏感性消融实验：
  对指定点云文件遍历 RANSAC threshold × DBSCAN (eps, min_samples) 的参数组合，
  记录不同参数对聚类结果和处理时间的影响。
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

def _add_project_root():
    root = Path(__file__).resolve().parent.parent.parent
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


# ---------------------------------------------------------------------------
# 复用 eval_segmentation 中的点云读取与几何分割
# ---------------------------------------------------------------------------

def _load_pcd(file_path: Path) -> Optional[np.ndarray]:
    """读取点云，返回 (N,3) float32，失败返回 None。"""
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
        return None
    except Exception as e:
        print(f"[eval_fps] 读取失败 {file_path}: {e}")
        return None


def _ransac_ground_simple(
    pts: np.ndarray,
    distance_threshold: float,
    iterations: int = 50,
) -> np.ndarray:
    """内置 RANSAC，返回 ground_mask (N,) bool。"""
    N = len(pts)
    best_mask  = np.zeros(N, dtype=bool)
    best_count = 0
    for _ in range(iterations):
        idx = np.random.choice(N, 3, replace=False)
        p0, p1, p2 = pts[idx[0]], pts[idx[1]], pts[idx[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        nlen = np.linalg.norm(normal)
        if nlen < 1e-6:
            continue
        normal /= nlen
        dists = np.abs(pts @ normal + (-np.dot(normal, p0)))
        mask = dists < distance_threshold
        count = int(np.sum(mask))
        if count > best_count:
            best_count = count
            best_mask  = mask
    return best_mask


def _dbscan_cluster(pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN 聚类，返回 labels (N,)。-1 为噪声。"""
    if len(pts) == 0:
        return np.array([], dtype=int)
    try:
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=eps, min_samples=min_samples,
                      algorithm="ball_tree", n_jobs=1).fit_predict(pts)
    except ImportError:
        pass
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        neighbors = tree.query_ball_point(pts, r=eps)
        N = len(pts)
        labels   = np.full(N, -1, dtype=int)
        visited  = np.zeros(N, dtype=bool)
        cid = 0
        for i in range(N):
            if visited[i]:
                continue
            visited[i] = True
            nb = neighbors[i]
            if len(nb) < min_samples:
                continue
            labels[i] = cid
            stack = list(nb)
            while stack:
                j = stack.pop()
                if not visited[j]:
                    visited[j] = True
                    nb2 = neighbors[j]
                    if len(nb2) >= min_samples:
                        stack.extend(nb2)
                if labels[j] == -1:
                    labels[j] = cid
            cid += 1
        return labels
    except Exception:
        return np.zeros(len(pts), dtype=int)


def _run_pipeline(
    pts: np.ndarray,
    ransac_threshold: float,
    dbscan_eps: float,
    dbscan_min_samples: int,
    cluster_min_points: int = 15,
) -> Dict:
    """
    运行完整处理链：RANSAC 地面分割 + DBSCAN 聚类。
    返回耗时（ms）和聚类统计。
    """
    t0 = time.perf_counter()

    # 1. RANSAC 地面分割
    ground_mask  = _ransac_ground_simple(pts, ransac_threshold)
    nonground    = pts[~ground_mask]

    # 2. DBSCAN 聚类
    cluster_count   = 0
    obstacle_count  = 0
    if len(nonground) > 0:
        labels = _dbscan_cluster(nonground, dbscan_eps, dbscan_min_samples)
        unique = set(labels) - {-1}
        cluster_count  = len(unique)
        obstacle_count = sum(
            1 for cid in unique
            if int(np.sum(labels == cid)) >= cluster_min_points
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "elapsed_ms":     round(elapsed_ms, 2),
        "cluster_count":  cluster_count,
        "obstacle_count": obstacle_count,
    }


# ---------------------------------------------------------------------------
# 点云降采样工具
# ---------------------------------------------------------------------------

def random_subsample(pts: np.ndarray, target_n: int) -> np.ndarray:
    """随机下采样到 target_n 个点，不足则原样返回。"""
    if len(pts) <= target_n:
        return pts
    idx = np.random.choice(len(pts), target_n, replace=False)
    return pts[idx]


# ---------------------------------------------------------------------------
# 实验一：FPS 对比实验
# ---------------------------------------------------------------------------

def run_fps_comparison(
    pts_full: np.ndarray,
    output_dir: Path,
    repeat: int = 5,
) -> pd.DataFrame:
    """
    测试不同点云规模下的 FPS，每种规模重复 repeat 次。
    """
    scales = [
        ("original",  None),
        ("50000pts",  50000),
        ("20000pts",  20000),
        ("10000pts",  10000),
    ]

    rows = []
    for sample_mode, n_pts in scales:
        pts = pts_full if n_pts is None else random_subsample(pts_full, n_pts)
        actual_n = len(pts)

        times_ms = []
        for r in range(repeat):
            np.random.seed(r)  # 使 RANSAC 随机性可复现
            try:
                res = _run_pipeline(pts,
                                     ransac_threshold=0.20,
                                     dbscan_eps=0.50,
                                     dbscan_min_samples=10)
                times_ms.append(res["elapsed_ms"])
            except Exception as e:
                print(f"[eval_fps]   ⚠ {sample_mode} 第{r+1}次异常: {e}")

        if times_ms:
            avg_ms = round(float(np.mean(times_ms)), 2)
            min_ms = round(float(np.min(times_ms)), 2)
            max_ms = round(float(np.max(times_ms)), 2)
            fps    = round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0.0
        else:
            avg_ms = min_ms = max_ms = fps = None

        rows.append({
            "point_count":   actual_n,
            "sample_mode":   sample_mode,
            "repeat_times":  len(times_ms),
            "avg_time_ms":   avg_ms,
            "min_time_ms":   min_ms,
            "max_time_ms":   max_ms,
            "fps":           fps,
        })
        print(f"[eval_fps]   {sample_mode:12s}  N={actual_n:6d}  "
              f"avg={avg_ms}ms  fps={fps}")

    df = pd.DataFrame(rows)
    csv_path = output_dir / "fps_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[eval_fps] 已保存: {csv_path}")
    return df


# ---------------------------------------------------------------------------
# 实验二：参数敏感性消融实验
# ---------------------------------------------------------------------------

def run_parameter_ablation(
    pts: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    """
    遍历参数组合，记录不同参数对聚类结果和性能的影响。
    """
    ransac_thresholds  = [0.10, 0.20, 0.30]
    dbscan_eps_values  = [0.30, 0.50, 0.70]
    dbscan_min_samples = [3,    5,    10]

    combos = list(itertools.product(
        ransac_thresholds,
        dbscan_eps_values,
        dbscan_min_samples,
    ))
    print(f"[eval_fps] 参数消融共 {len(combos)} 组...")

    rows = []
    for rt, eps, min_s in combos:
        times_ms = []
        results  = []
        for seed in range(3):  # 每组重复 3 次
            np.random.seed(seed)
            try:
                res = _run_pipeline(pts, rt, eps, min_s)
                times_ms.append(res["elapsed_ms"])
                results.append(res)
            except Exception as e:
                print(f"[eval_fps]   ⚠ 参数组合 rt={rt} eps={eps} min_s={min_s} 异常: {e}")

        if times_ms:
            avg_ms = round(float(np.mean(times_ms)), 2)
            fps    = round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0.0
            cluster_count  = int(np.mean([r["cluster_count"]  for r in results]))
            obstacle_count = int(np.mean([r["obstacle_count"] for r in results]))

            # 稳定性备注：各次 cluster_count 方差
            cc_std = float(np.std([r["cluster_count"] for r in results]))
            if cc_std < 1.0:
                stability_note = "稳定"
            elif cc_std < 3.0:
                stability_note = "轻微波动"
            else:
                stability_note = "波动较大"
        else:
            avg_ms = fps = None
            cluster_count = obstacle_count = None
            stability_note = "评估失败"

        rows.append({
            "ransac_threshold":    rt,
            "dbscan_eps":          eps,
            "dbscan_min_samples":  min_s,
            "cluster_count":       cluster_count,
            "obstacle_count":      obstacle_count,
            "avg_time_ms":         avg_ms,
            "fps":                 fps,
            "stability_note":      stability_note,
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "parameter_ablation.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[eval_fps] 已保存: {csv_path}（共 {len(df)} 组）")
    return df


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace, output_dir: Path):
    """
    FPS 对比 + 参数消融实验入口。
    自动从 data_dir 中选取第一个 .bin / .pcd 文件作为测试样本。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # 查找点云文件
    pts_full = None
    used_file = None
    for ext in ("*.bin", "*.pcd", "*.ply"):
        candidates = sorted(data_dir.rglob(ext))
        for fp in candidates:
            pts = _load_pcd(fp)
            if pts is not None and len(pts) > 1000:
                pts_full   = pts
                used_file  = fp
                break
        if pts_full is not None:
            break

    if pts_full is None:
        print(f"[eval_fps] ⚠ 未在 {data_dir} 找到有效点云（>1000 点），使用随机点云代替")
        np.random.seed(42)
        pts_full  = np.random.randn(30000, 3).astype(np.float32)
        pts_full[:, 2] *= 0.1   # 模拟地面较平
        used_file = Path("synthetic_random")

    print(f"[eval_fps] 使用点云: {used_file}  N={len(pts_full)}")

    # 实验一：FPS 对比
    print("\n[eval_fps] === 实验一：FPS 对比 ===")
    df_fps = run_fps_comparison(pts_full, output_dir, repeat=5)

    # 实验二：参数消融
    # 为节省时间，消融实验限制最大 20000 点
    pts_ablation = random_subsample(pts_full, 20000)
    print(f"\n[eval_fps] === 实验二：参数消融（N={len(pts_ablation)}）===")
    df_abl = run_parameter_ablation(pts_ablation, output_dir)

    return df_fps, df_abl


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FPS 对比 + 参数消融实验模块")
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--output_dir", default="outputs/experiments")
    args = parser.parse_args()

    _add_project_root()
    out = Path(args.output_dir)
    main(args, out)
