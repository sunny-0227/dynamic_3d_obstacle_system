#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_to_json.py — OpenPCDet 独立推理脚本（WSL2 Ubuntu 运行环境）

功能：
  对指定目录（或单个文件）中的点云文件执行 OpenPCDet 3D 目标检测推理，
  将所有帧的检测结果写入一个 JSON 文件。

输出 JSON 结构：
  [
    {
      "file":   "xxx.bin",        # 文件名（不含目录）
      "boxes":  [[x,y,z,l,w,h,yaw], ...],   # 每框 7 个值
      "labels": [1, 2, ...],      # 类别 id（1-based，与 class_names 对应）
      "scores": [0.92, ...]       # 置信度
    },
    ...
  ]

依赖：
  pip install torch open3d
  # OpenPCDet 需在当前 Python 环境已安装（cd OpenPCDet && pip install -e .）

用法示例：
  python infer_to_json.py \\
      --cfg_file  OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml \\
      --ckpt      checkpoints/pointpillar_7728.pth \\
      --data_path data/kitti/testing/velodyne \\
      --ext       .bin \\
      --out_json  results/detections.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────
# 工具函数：打印错误并退出
# ──────────────────────────────────────────────────────────────

def _fatal(msg: str, exc: Optional[BaseException] = None) -> None:
    """打印清晰的错误信息后以非 0 退出码终止进程。"""
    print(f"\n[ERROR] {msg}", file=sys.stderr)
    if exc is not None:
        print(f"        原因：{type(exc).__name__}: {exc}", file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenPCDet 独立推理脚本，结果写入 JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="OpenPCDet 模型配置文件路径（yaml），如 cfgs/kitti_models/pointpillar.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="模型权重 checkpoint 路径（.pth）",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="点云数据路径：可以是单个文件，也可以是包含点云文件的目录",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".bin",
        help="点云文件扩展名，默认 .bin（也支持 .npy / .pcd）",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="detections.json",
        help="输出 JSON 文件路径，默认 detections.json",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# OpenPCDet 依赖导入（提前校验，给出清晰错误提示）
# ──────────────────────────────────────────────────────────────

def _import_openpcdet():
    """
    导入 OpenPCDet 核心组件，并在缺失时给出详细的安装提示。
    返回：(cfg, cfg_from_yaml_file, build_network, load_data_to_gpu, DemoDataset)
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        _fatal(
            "未找到 PyTorch，请先安装：pip install torch",
            e,
        )

    try:
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
    except ImportError as e:
        _fatal(
            "未找到 pcdet 包。\n"
            "        请在 OpenPCDet 项目根目录执行：\n"
            "            cd OpenPCDet && pip install -e .",
            e,
        )

    return cfg, cfg_from_yaml_file, build_network, load_data_to_gpu, DatasetTemplate, common_utils


# ──────────────────────────────────────────────────────────────
# DemoDataset：参考 OpenPCDet tools/demo.py 官方实现
# ──────────────────────────────────────────────────────────────

def _build_demo_dataset(DatasetTemplate, dataset_cfg, class_names: List[str], file_list: List[Path]):
    """
    构建与 OpenPCDet 官方 demo.py 一致的 DemoDataset 子类。

    官方 demo.py 的 DemoDataset 继承自 DatasetTemplate，
    只需实现 __len__ / __getitem__ / get_infos / generate_prediction_dicts。
    """

    class DemoDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names, training=False, root_path=None, logger=None):
            super().__init__(
                dataset_cfg=dataset_cfg,
                class_names=class_names,
                training=training,
                root_path=root_path,
                logger=logger,
            )
            self.file_list = file_list

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, index):
            """
            读取单帧点云并返回 data_dict。
            支持格式：
              .bin  — KITTI/nuScenes 格式，float32，列数 4（x y z intensity）或 5
              .npy  — numpy 存储，自动 reshape 到 (N, C)
              .pcd  — 通过 open3d 读取，仅使用 xyz
            """
            fpath = self.file_list[index]
            ext = fpath.suffix.lower()

            try:
                if ext == ".bin":
                    # KITTI 格式：float32，每点 4 列（x, y, z, intensity）
                    points = np.fromfile(str(fpath), dtype=np.float32)
                    # 尝试 4 列，若余数不为 0 则尝试 5 列
                    for ncols in (4, 5, 3):
                        if points.size % ncols == 0:
                            points = points.reshape(-1, ncols)
                            break
                    else:
                        raise ValueError(
                            f"无法将 .bin 文件解析为 3/4/5 列，总元素数={points.size}"
                        )

                elif ext == ".npy":
                    points = np.load(str(fpath)).astype(np.float32)
                    if points.ndim == 1:
                        # 猜测列数
                        for ncols in (4, 5, 3):
                            if points.size % ncols == 0:
                                points = points.reshape(-1, ncols)
                                break

                elif ext in (".pcd", ".ply"):
                    try:
                        import open3d as o3d
                    except ImportError:
                        raise ImportError(
                            "读取 .pcd/.ply 需要 open3d，请安装：pip install open3d"
                        )
                    pcd = o3d.io.read_point_cloud(str(fpath))
                    xyz = np.asarray(pcd.points, dtype=np.float32)
                    # 补充强度列（填 0），凑成 4 列
                    intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
                    points = np.hstack([xyz, intensity])

                else:
                    raise ValueError(f"不支持的点云格式：{ext}，支持 .bin / .npy / .pcd / .ply")

            except Exception as exc:
                raise RuntimeError(f"读取点云文件失败：{fpath}\n原因：{exc}") from exc

            # OpenPCDet DatasetTemplate 要求 points 至少有 4 列（x y z intensity），
            # 若只有 3 列则补 0
            if points.shape[1] == 3:
                intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack([points, intensity])

            input_dict = {
                "points": points,
                "frame_id": index,
            }
            data_dict = self.prepare_data(data_dict=input_dict)
            return data_dict

    return DemoDataset


# ──────────────────────────────────────────────────────────────
# 推理核心
# ──────────────────────────────────────────────────────────────

def run_inference(args: argparse.Namespace) -> List[Dict]:
    """
    主推理流程，完全对齐 OpenPCDet 官方 tools/demo.py 逻辑：
      1. 加载模型配置
      2. 构建模型 + 加载 checkpoint
      3. 遍历点云文件，逐帧推理
      4. 汇总结果
    """
    import torch

    cfg, cfg_from_yaml_file, build_network, load_data_to_gpu, DatasetTemplate, common_utils = (
        _import_openpcdet()
    )

    # ── 1. 检查路径 ────────────────────────────────────────────
    cfg_path = Path(args.cfg_file)
    ckpt_path = Path(args.ckpt)
    data_path = Path(args.data_path)
    out_json  = Path(args.out_json)

    if not cfg_path.exists():
        _fatal(f"配置文件不存在：{cfg_path}")
    if not ckpt_path.exists():
        _fatal(f"Checkpoint 文件不存在：{ckpt_path}")
    if not data_path.exists():
        _fatal(f"数据路径不存在：{data_path}")

    # ── 2. 收集点云文件列表 ─────────────────────────────────────
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"
    if data_path.is_file():
        file_list = [data_path]
    else:
        file_list = sorted(data_path.glob(f"*{ext}"))

    if not file_list:
        _fatal(
            f"在 {data_path} 中未找到任何 {ext} 文件，\n"
            f"        请检查 --data_path 和 --ext 参数。"
        )

    print(f"[INFO] 找到 {len(file_list)} 个点云文件，开始推理…")

    # ── 3. 加载模型配置 ─────────────────────────────────────────
    try:
        cfg_from_yaml_file(str(cfg_path), cfg)
    except Exception as e:
        _fatal(f"加载模型配置失败：{cfg_path}", e)

    # 从配置中读取类别名
    class_names: List[str] = cfg.CLASS_NAMES

    # 构建日志（OpenPCDet 内部使用）
    logger = common_utils.create_logger()
    logger.info(f"类别名称：{class_names}")

    # ── 4. 构建 DemoDataset ──────────────────────────────────────
    DemoDataset = _build_demo_dataset(DatasetTemplate, cfg.DATA_CONFIG, class_names, file_list)

    try:
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=class_names,
            training=False,
            root_path=data_path if data_path.is_dir() else data_path.parent,
            logger=logger,
        )
    except Exception as e:
        _fatal("构建 DemoDataset 失败", e)

    # ── 5. 构建模型并加载权重 ────────────────────────────────────
    try:
        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(class_names),
            dataset=demo_dataset,
        )
    except Exception as e:
        _fatal("build_network 失败，请检查配置文件与 pcdet 版本是否匹配", e)

    try:
        model.load_params_from_file(
            filename=str(ckpt_path),
            logger=logger,
            to_cpu=True,  # WSL2 无 CUDA 时保持兼容
        )
    except Exception as e:
        _fatal(f"加载权重失败：{ckpt_path}", e)

    # 决定推理设备
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 推理设备：{device_str}")
    device = torch.device(device_str)
    model = model.to(device)
    model.eval()

    # ── 6. 逐帧推理 ─────────────────────────────────────────────
    results: List[Dict] = []

    with torch.no_grad():
        for idx in range(len(demo_dataset)):
            fpath = file_list[idx]
            t0 = time.perf_counter()

            # 读取并预处理当前帧
            try:
                data_dict = demo_dataset[idx]
            except Exception as e:
                print(
                    f"[WARN] 读取第 {idx+1}/{len(file_list)} 帧失败，跳过：{fpath.name}\n"
                    f"       原因：{e}",
                    file=sys.stderr,
                )
                results.append({
                    "file":   fpath.name,
                    "boxes":  [],
                    "labels": [],
                    "scores": [],
                    "error":  str(e),
                })
                continue

            # collate_batch：将单帧 dict 包装成 batch（与官方 demo.py 一致）
            try:
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
            except Exception as e:
                _fatal(f"数据预处理（collate / GPU 上传）失败：{fpath.name}", e)

            # 前向推理
            try:
                pred_dicts, _ = model.forward(data_dict)
            except Exception as e:
                print(
                    f"[WARN] 推理失败，跳过第 {idx+1}/{len(file_list)} 帧：{fpath.name}\n"
                    f"       原因：{e}",
                    file=sys.stderr,
                )
                results.append({
                    "file":   fpath.name,
                    "boxes":  [],
                    "labels": [],
                    "scores": [],
                    "error":  str(e),
                })
                continue

            # 提取预测结果（batch size = 1，取 [0]）
            pred = pred_dicts[0]
            boxes_tensor  = pred.get("pred_boxes",  None)
            labels_tensor = pred.get("pred_labels", None)
            scores_tensor = pred.get("pred_scores", None)

            if boxes_tensor is None or boxes_tensor.shape[0] == 0:
                # 本帧无检测结果
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(
                    f"[INFO] 帧 {idx+1:>4d}/{len(file_list)} | "
                    f"{fpath.name:40s} | 检测到 0 个目标 | {elapsed_ms:.1f} ms"
                )
                results.append({
                    "file":   fpath.name,
                    "boxes":  [],
                    "labels": [],
                    "scores": [],
                })
                continue

            # 将 tensor 转为 Python 列表（JSON 可序列化）
            boxes_np  = boxes_tensor.cpu().numpy()   # (M, 7)  [x,y,z,l,w,h,yaw]
            labels_np = labels_tensor.cpu().numpy()  # (M,)   1-based 类别 id
            scores_np = scores_tensor.cpu().numpy()  # (M,)

            boxes_list  = boxes_np.tolist()
            labels_list = [int(v) for v in labels_np]
            scores_list = [round(float(v), 6) for v in scores_np]

            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(
                f"[INFO] 帧 {idx+1:>4d}/{len(file_list)} | "
                f"{fpath.name:40s} | 检测到 {len(boxes_list):>3d} 个目标 | {elapsed_ms:.1f} ms"
            )

            results.append({
                "file":   fpath.name,
                "boxes":  boxes_list,
                "labels": labels_list,
                "scores": scores_list,
            })

    return results


# ──────────────────────────────────────────────────────────────
# 写出 JSON
# ──────────────────────────────────────────────────────────────

def _write_json(results: List[Dict], out_path: Path) -> None:
    """将结果列表写入 JSON 文件，遇到错误则打印并退出。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _fatal(f"写入 JSON 文件失败：{out_path}", e)

    print(f"\n[INFO] 推理完成，结果已写入：{out_path.resolve()}")
    print(f"[INFO] 共 {len(results)} 帧，"
          f"检测到目标的帧数：{sum(1 for r in results if r.get('boxes'))}")


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("  OpenPCDet 独立推理脚本")
    print(f"  cfg_file  : {args.cfg_file}")
    print(f"  ckpt      : {args.ckpt}")
    print(f"  data_path : {args.data_path}")
    print(f"  ext       : {args.ext}")
    print(f"  out_json  : {args.out_json}")
    print("=" * 60)

    results = run_inference(args)
    _write_json(results, Path(args.out_json))


if __name__ == "__main__":
    main()
